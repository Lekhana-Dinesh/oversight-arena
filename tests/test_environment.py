"""Tests for the core Oversight Arena environment state machine."""

from __future__ import annotations

import json

import pytest

from oversight_arena.data_generator import Difficulty, Domain
from oversight_arena.environment import (
    EnvironmentStateError,
    InvalidEnvironmentAction,
    OversightArenaEnv,
)
from oversight_arena.grader import grade_episode
from oversight_arena.models import ActionKind, ErrorCategory, FlaggedAnswer, OversightAction


def accept_action() -> OversightAction:
    """Build an accept-current-answer action."""

    return OversightAction(action=ActionKind.ACCEPT_ALL)


def flag_action(answer_id: str, category: ErrorCategory, rationale: str) -> OversightAction:
    """Build a flag-current-answer action."""

    return OversightAction(
        action=ActionKind.FLAG_ERRORS,
        flags=(
            FlaggedAnswer(
                answer_id=answer_id,
                error_category=category,
                rationale=rationale,
            ),
        ),
    )


def action_for_current_truth(env: OversightArenaEnv) -> OversightAction:
    """Build the correct action for the environment's current answer."""

    episode = env.generated_episode()
    current_answer_id = env.current_observation().worker_answers[0].answer_id
    truth = next(item for item in episode.worker_truths if item.answer_id == current_answer_id)
    if truth.is_correct:
        return accept_action()
    return flag_action(
        answer_id=truth.answer_id,
        category=truth.error_category,
        rationale=episode.evidence_for(truth.answer_id).citation_text(),
    )


def assert_public_observation_has_no_hidden_truth(payload_source: object) -> None:
    """Assert an observation-shaped object does not serialize hidden metadata."""

    serialized_payload = json.dumps(payload_source.model_dump(mode="json"))

    assert "is_correct" not in serialized_payload
    assert "expected_answer" not in serialized_payload
    assert "reviewer_note" not in serialized_payload
    assert "evidence" not in serialized_payload
    assert "error_category" not in serialized_payload


def test_reset_is_deterministic_for_explicit_parameters() -> None:
    """Reset with identical parameters produces identical public observations."""

    first = OversightArenaEnv()
    second = OversightArenaEnv()

    first_observation = first.reset(
        seed=101,
        domain=Domain.FINANCE,
        difficulty=Difficulty.HARD,
        error_count=2,
    )
    second_observation = second.reset(
        seed=101,
        domain=Domain.FINANCE,
        difficulty=Difficulty.HARD,
        error_count=2,
    )

    assert first_observation == second_observation
    assert first.generated_episode() == second.generated_episode()
    assert first_observation.worker_answers == (
        first.generated_episode().worker_truths[0].to_public_answer(),
    )


def test_reset_and_step_observations_do_not_leak_hidden_truth() -> None:
    """Reset and non-terminal step observations remain public-only."""

    env = OversightArenaEnv()
    reset_observation = env.reset(
        seed=103,
        domain=Domain.LOGISTICS,
        difficulty=Difficulty.MEDIUM,
    )

    assert_public_observation_has_no_hidden_truth(reset_observation)

    step_result = env.step(action_for_current_truth(env))

    assert step_result.observation is not None
    assert_public_observation_has_no_hidden_truth(step_result.observation)


def test_valid_step_progression_produces_final_grade() -> None:
    """Reviewing one answer at a time reaches a deterministic terminal grade."""

    env = OversightArenaEnv()
    first_observation = env.reset(
        seed=107,
        domain=Domain.RETAIL,
        difficulty=Difficulty.HARD,
        error_count=2,
    )
    total_count = len(env.generated_episode().worker_truths)

    assert first_observation.turn_index == 0
    assert len(first_observation.worker_answers) == 1

    results = []
    while not env.done:
        result = env.step(action_for_current_truth(env))
        results.append(result)
        assert result.answer_grade is not None
        assert result.reviewed_count <= result.total_count == total_count

    final_result = results[-1]

    assert final_result.done
    assert final_result.observation is None
    assert final_result.final_grade is not None
    assert env.final_grade == final_result.final_grade
    assert final_result.final_grade.final_score == pytest.approx(1.0)
    assert len(env.reviewed_answer_ids) == total_count


def test_duplicate_or_repeated_review_is_rejected() -> None:
    """Actions for an already reviewed answer cannot be applied on a later turn."""

    env = OversightArenaEnv()
    observation = env.reset(seed=109, domain=Domain.FINANCE, difficulty=Difficulty.EASY)
    reviewed_answer_id = observation.worker_answers[0].answer_id

    env.step(accept_action())

    with pytest.raises(InvalidEnvironmentAction, match="current answer_id"):
        env.step(
            flag_action(
                answer_id=reviewed_answer_id,
                category=ErrorCategory.NUMERIC_MISMATCH,
                rationale="finance-invoice-001 invoice_total",
            )
        )


def test_unknown_answer_id_is_rejected() -> None:
    """Flags must target the current public answer ID."""

    env = OversightArenaEnv()
    env.reset(seed=113, domain=Domain.RETAIL, difficulty=Difficulty.EASY)

    with pytest.raises(InvalidEnvironmentAction, match="current answer_id"):
        env.step(
            flag_action(
                answer_id="missing-answer",
                category=ErrorCategory.ENTITY_MISMATCH,
                rationale="retail-order-001 item",
            )
        )


def test_duplicate_flags_are_rejected_even_if_action_validation_is_bypassed() -> None:
    """Environment validates duplicate flags even for constructed action objects."""

    env = OversightArenaEnv()
    observation = env.reset(seed=117, domain=Domain.LOGISTICS, difficulty=Difficulty.EASY)
    current_answer_id = observation.worker_answers[0].answer_id
    duplicate_flag = FlaggedAnswer(
        answer_id=current_answer_id,
        error_category=ErrorCategory.ENTITY_MISMATCH,
        rationale="logistics-shipment-001 destination",
    )
    action = OversightAction.model_construct(
        action=ActionKind.FLAG_ERRORS,
        flags=(duplicate_flag, duplicate_flag),
    )

    with pytest.raises(InvalidEnvironmentAction, match="duplicate flags"):
        env.step(action)


def test_step_before_reset_and_after_done_are_explicit_errors() -> None:
    """Invalid reset/step ordering raises deterministic state errors."""

    env = OversightArenaEnv()

    with pytest.raises(EnvironmentStateError, match="reset"):
        env.step(accept_action())

    env.reset(seed=121, domain=Domain.FINANCE, difficulty=Difficulty.EASY, error_count=0)
    while not env.done:
        env.step(accept_action())

    with pytest.raises(EnvironmentStateError, match="after episode completion"):
        env.step(accept_action())

    with pytest.raises(EnvironmentStateError, match="after episode completion"):
        env.current_observation()


def test_zero_error_episode_accept_all_scores_perfectly() -> None:
    """Zero-error episodes terminate cleanly when every answer is accepted."""

    env = OversightArenaEnv()
    env.reset(seed=127, domain=Domain.RETAIL, difficulty=Difficulty.MEDIUM, error_count=0)
    result = None

    while not env.done:
        result = env.step(accept_action())

    assert result is not None
    assert result.final_grade is not None
    assert result.final_grade.total_incorrect == 0
    assert result.final_grade.total_flags == 0
    assert result.final_grade.final_score == pytest.approx(1.0)


def test_final_grade_matches_current_grader_for_aggregated_decisions() -> None:
    """The environment delegates final scoring to the current grader semantics."""

    env = OversightArenaEnv()
    env.reset(seed=131, domain=Domain.LOGISTICS, difficulty=Difficulty.HARD, error_count=2)
    episode = env.generated_episode()
    submitted_flags: list[FlaggedAnswer] = []

    while not env.done:
        action = action_for_current_truth(env)
        submitted_flags.extend(action.flags)
        result = env.step(action)

    expected_action = OversightAction(action=ActionKind.FLAG_ERRORS, flags=tuple(submitted_flags))
    expected_grade = grade_episode(
        expected_action,
        episode.worker_truths,
        episode.source_records,
    )

    assert result.final_grade == expected_grade
