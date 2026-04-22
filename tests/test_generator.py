"""Tests for deterministic generated Oversight Arena episodes."""

from __future__ import annotations

import json

import pytest

from oversight_arena.data_generator import Difficulty, Domain, GeneratedEpisode, generate_episode
from oversight_arena.grader import grade_episode
from oversight_arena.models import ActionKind, ErrorCategory, FlaggedAnswer, OversightAction


def test_generation_is_deterministic_for_fixed_parameters() -> None:
    """Same seed and parameters produce identical generated episodes."""

    first = generate_episode(seed=101, domain=Domain.FINANCE, difficulty=Difficulty.HARD)
    second = generate_episode(seed=101, domain=Domain.FINANCE, difficulty=Difficulty.HARD)
    different = generate_episode(seed=102, domain=Domain.FINANCE, difficulty=Difficulty.HARD)

    assert first == second
    assert first != different


def test_generated_ids_are_unique_and_stable() -> None:
    """Generated public and internal IDs are stable and unique."""

    episode = generate_episode(seed=7, domain=Domain.LOGISTICS, difficulty=Difficulty.MEDIUM)
    repeat = generate_episode(seed=7, domain=Domain.LOGISTICS, difficulty=Difficulty.MEDIUM)

    source_ids = [record.record_id for record in episode.source_records]
    answer_ids = [truth.answer_id for truth in episode.worker_truths]
    evidence_ids = [evidence.answer_id for evidence in episode.evidence]

    assert source_ids == [record.record_id for record in repeat.source_records]
    assert answer_ids == [truth.answer_id for truth in repeat.worker_truths]
    assert len(source_ids) == len(set(source_ids))
    assert len(answer_ids) == len(set(answer_ids))
    assert evidence_ids == answer_ids


def test_generated_episode_is_internally_consistent() -> None:
    """Truth records and evidence references point to generated source data."""

    episode = generate_episode(seed=9, domain=Domain.RETAIL, difficulty=Difficulty.EXPERT)
    source_fields_by_id = {
        record.record_id: set(record.fields) for record in episode.source_records
    }

    assert isinstance(episode, GeneratedEpisode)
    assert episode.manifest().to_observation() == episode.to_observation()
    assert episode.public_worker_answers() == episode.to_observation().worker_answers

    for truth in episode.worker_truths:
        evidence = episode.evidence_for(truth.answer_id)
        assert evidence.source_record_ids == truth.source_record_ids
        assert evidence.references
        for reference in evidence.references:
            assert reference.record_id in source_fields_by_id
            assert reference.field_name in source_fields_by_id[reference.record_id]

        if truth.is_correct:
            assert truth.error_category is None
            assert truth.expected_answer is None
            assert evidence.error_category is None
            assert evidence.expected_answer == truth.answer
        else:
            assert truth.error_category is not None
            assert truth.expected_answer == evidence.expected_answer
            assert truth.answer != truth.expected_answer


def test_public_projection_does_not_leak_hidden_truth_or_evidence() -> None:
    """Public observations expose source data and worker answers only."""

    episode = generate_episode(seed=13, domain=Domain.FINANCE, difficulty=Difficulty.MEDIUM)

    public_payload = episode.to_observation().model_dump(mode="json")
    serialized_payload = json.dumps(public_payload)

    assert "is_correct" not in serialized_payload
    assert "expected_answer" not in serialized_payload
    assert "reviewer_note" not in serialized_payload
    assert "evidence" not in serialized_payload
    assert "error_category" not in serialized_payload


def test_explicit_evidence_metadata_supports_current_reasoning_citations() -> None:
    """Evidence citation text can drive a perfect current-grader rationale."""

    episode = generate_episode(
        seed=23,
        domain=Domain.LOGISTICS,
        difficulty=Difficulty.HARD,
        error_count=2,
    )
    flags = tuple(
        FlaggedAnswer(
            answer_id=truth.answer_id,
            error_category=truth.error_category,
            rationale=episode.evidence_for(truth.answer_id).citation_text(),
        )
        for truth in episode.worker_truths
        if not truth.is_correct
    )

    grade = grade_episode(
        OversightAction(action=ActionKind.FLAG_ERRORS, flags=flags),
        episode.worker_truths,
        episode.source_records,
    )

    assert grade.true_positive_flags == 2
    assert grade.precision_score == pytest.approx(1.0)
    assert grade.recall_score == pytest.approx(1.0)
    assert grade.reasoning_quality == pytest.approx(1.0)
    assert grade.final_score == pytest.approx(1.0)


def test_zero_error_episode_generation_is_supported() -> None:
    """Generator can produce episodes with no injected worker errors."""

    episode = generate_episode(
        seed=29,
        domain=Domain.RETAIL,
        difficulty=Difficulty.EASY,
        error_count=0,
    )
    grade = grade_episode(
        OversightAction(action=ActionKind.ACCEPT_ALL),
        episode.worker_truths,
        episode.source_records,
    )

    assert all(truth.is_correct for truth in episode.worker_truths)
    assert all(evidence.error_category is None for evidence in episode.evidence)
    assert grade.final_score == pytest.approx(1.0)


def test_all_domains_generate_default_correct_and_incorrect_answers() -> None:
    """Every supported domain produces both correct and incorrect answers by default."""

    for domain in Domain:
        episode = generate_episode(seed=31, domain=domain, difficulty=Difficulty.MEDIUM)
        correctness = {truth.is_correct for truth in episode.worker_truths}

        assert correctness == {False, True}
        assert {record.record_type for record in episode.source_records}


def test_difficulty_changes_size_and_evidence_complexity() -> None:
    """Difficulty affects record count, answer count, and evidence complexity."""

    easy = generate_episode(seed=37, domain=Domain.FINANCE, difficulty=Difficulty.EASY)
    medium = generate_episode(seed=37, domain=Domain.FINANCE, difficulty=Difficulty.MEDIUM)
    hard = generate_episode(seed=37, domain=Domain.FINANCE, difficulty=Difficulty.HARD)
    expert = generate_episode(seed=37, domain=Domain.FINANCE, difficulty=Difficulty.EXPERT)

    assert [len(item.source_records) for item in (easy, medium, hard, expert)] == [2, 3, 4, 5]
    assert [len(item.worker_truths) for item in (easy, medium, hard, expert)] == [3, 4, 5, 6]
    assert any(len(evidence.references) > 1 for evidence in hard.evidence)
    assert any(len(evidence.references) > 1 for evidence in expert.evidence)


def test_category_mismatch_possibility_is_compatible_with_current_grader() -> None:
    """Generated truth categories allow deterministic category-mismatch scoring."""

    episode = generate_episode(
        seed=41,
        domain=Domain.FINANCE,
        difficulty=Difficulty.EXPERT,
        error_count=1,
    )
    incorrect_truth = next(truth for truth in episode.worker_truths if not truth.is_correct)
    mismatched_category = next(
        category for category in ErrorCategory if category is not incorrect_truth.error_category
    )
    grade = grade_episode(
        OversightAction(
            action=ActionKind.FLAG_ERRORS,
            flags=(
                FlaggedAnswer(
                    answer_id=incorrect_truth.answer_id,
                    error_category=mismatched_category,
                    rationale=episode.evidence_for(incorrect_truth.answer_id).citation_text(),
                ),
            ),
        ),
        episode.worker_truths,
        episode.source_records,
    )

    assert grade.true_positive_flags == 1
    assert grade.category_mismatch_flags == 1
    assert grade.reasoning_quality == pytest.approx(0.0)
    assert grade.efficiency_score < 1.0


def test_error_count_validation() -> None:
    """Generator rejects impossible controlled error counts."""

    with pytest.raises(ValueError, match="non-negative"):
        generate_episode(seed=1, error_count=-1)

    with pytest.raises(ValueError, match="no greater"):
        generate_episode(seed=1, difficulty=Difficulty.EASY, error_count=4)
