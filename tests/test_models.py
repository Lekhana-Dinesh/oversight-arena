"""Contract tests for Oversight Arena schemas."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from oversight_arena.models import (
    ACTION_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    ActionKind,
    EpisodeManifest,
    ErrorCategory,
    FlaggedAnswer,
    OversightAction,
    OversightObservation,
    SourceRecord,
    WorkerAnswer,
    WorkerAnswerTruth,
)


def make_source_record(record_id: str = "source-1") -> SourceRecord:
    """Build a minimal valid public source record for tests."""

    return SourceRecord(
        record_id=record_id,
        record_type="invoice",
        fields={"vendor": "Northwind", "units": 18, "approved": True},
    )


def make_worker_truth(
    answer_id: str = "answer-1",
    *,
    is_correct: bool = False,
    source_record_ids: tuple[str, ...] = ("source-1",),
) -> WorkerAnswerTruth:
    """Build a valid internal answer truth record for tests."""

    if is_correct:
        return WorkerAnswerTruth(
            answer_id=answer_id,
            question="How many units were approved?",
            answer="18 units",
            source_record_ids=source_record_ids,
            is_correct=True,
        )

    return WorkerAnswerTruth(
        answer_id=answer_id,
        question="How many units were approved?",
        answer="19 units",
        source_record_ids=source_record_ids,
        is_correct=False,
        expected_answer="18 units",
        error_category=ErrorCategory.NUMERIC_MISMATCH,
        reviewer_note="The source record states 18 approved units.",
    )


def test_manifest_projects_hidden_truth_to_public_observation() -> None:
    """Internal truth metadata must not appear in public observation dumps."""

    manifest = EpisodeManifest(
        episode_id="episode-1",
        source_records=(make_source_record(),),
        worker_answers=(make_worker_truth(),),
    )

    observation = manifest.to_observation()

    assert isinstance(observation, OversightObservation)
    assert observation.schema_version == OBSERVATION_SCHEMA_VERSION
    assert observation.worker_answers[0] == WorkerAnswer(
        answer_id="answer-1",
        question="How many units were approved?",
        answer="19 units",
        source_record_ids=("source-1",),
    )

    public_payload = observation.model_dump(mode="json")
    serialized_payload = json.dumps(public_payload)

    assert "is_correct" not in serialized_payload
    assert "expected_answer" not in serialized_payload
    assert "reviewer_note" not in serialized_payload
    assert "numeric_mismatch" not in serialized_payload


def test_public_observation_rejects_duplicate_ids_and_unknown_references() -> None:
    """Public observations enforce unique IDs and known source references."""

    source = make_source_record()

    with pytest.raises(ValidationError):
        OversightObservation(
            schema_version="oversight_arena.observation.v2",
            episode_id="episode-1",
            source_records=(source,),
            worker_answers=(make_worker_truth().to_public_answer(),),
        )

    with pytest.raises(ValidationError, match="unique record_id"):
        OversightObservation(
            episode_id="episode-1",
            source_records=(source, source),
            worker_answers=(make_worker_truth().to_public_answer(),),
        )

    with pytest.raises(ValidationError, match="unknown source records"):
        OversightObservation(
            episode_id="episode-1",
            source_records=(source,),
            worker_answers=(
                WorkerAnswer(
                    answer_id="answer-1",
                    question="How many units were approved?",
                    answer="19 units",
                    source_record_ids=("missing-source",),
                ),
            ),
        )

    with pytest.raises(ValidationError, match="unique answer_id"):
        OversightObservation(
            episode_id="episode-1",
            source_records=(source,),
            worker_answers=(
                make_worker_truth("answer-1").to_public_answer(),
                make_worker_truth("answer-1").to_public_answer(),
            ),
        )


def test_internal_manifest_enforces_truth_consistency() -> None:
    """Hidden truth records reject inconsistent correctness metadata."""

    with pytest.raises(ValueError, match="must declare an error_category"):
        WorkerAnswerTruth(
            answer_id="answer-1",
            question="How many units were approved?",
            answer="19 units",
            source_record_ids=("source-1",),
            is_correct=False,
            expected_answer="18 units",
        )

    with pytest.raises(ValueError, match="non-empty expected_answer"):
        WorkerAnswerTruth(
            answer_id="answer-1",
            question="How many units were approved?",
            answer="19 units",
            source_record_ids=("source-1",),
            is_correct=False,
            error_category=ErrorCategory.NUMERIC_MISMATCH,
        )

    with pytest.raises(ValueError, match="must not declare an error_category"):
        WorkerAnswerTruth(
            answer_id="answer-1",
            question="How many units were approved?",
            answer="18 units",
            source_record_ids=("source-1",),
            is_correct=True,
            error_category=ErrorCategory.NUMERIC_MISMATCH,
        )


def test_action_accept_all_contract() -> None:
    """Accept-all actions are valid only when no answers are flagged."""

    action = OversightAction.model_validate({"action": "accept_all"})

    assert action.schema_version == ACTION_SCHEMA_VERSION
    assert action.action is ActionKind.ACCEPT_ALL
    assert action.flags == ()

    with pytest.raises(ValidationError, match="must not include flags"):
        OversightAction.model_validate(
            {
                "action": "accept_all",
                "flags": [
                    {
                        "answer_id": "answer-1",
                        "error_category": "numeric_mismatch",
                        "rationale": "The total does not match the source.",
                    }
                ],
            }
        )


def test_action_flag_errors_contract() -> None:
    """Flag-errors actions require unique flagged answer IDs."""

    action = OversightAction(
        action=ActionKind.FLAG_ERRORS,
        flags=(
            FlaggedAnswer(
                answer_id="answer-1",
                error_category=ErrorCategory.NUMERIC_MISMATCH,
                rationale="The worker reported 19 units, but the source record says 18.",
            ),
        ),
    )

    assert action.model_dump(mode="json") == {
        "schema_version": ACTION_SCHEMA_VERSION,
        "action": "flag_errors",
        "flags": [
            {
                "answer_id": "answer-1",
                "error_category": "numeric_mismatch",
                "rationale": "The worker reported 19 units, but the source record says 18.",
            }
        ],
    }

    with pytest.raises(ValidationError, match="at least one flag"):
        OversightAction.model_validate({"action": "flag_errors", "flags": []})

    with pytest.raises(ValidationError, match="duplicate answer_id"):
        OversightAction.model_validate(
            {
                "action": "flag_errors",
                "flags": [
                    {
                        "answer_id": "answer-1",
                        "error_category": "numeric_mismatch",
                        "rationale": "The total does not match the source.",
                    },
                    {
                        "answer_id": "answer-1",
                        "error_category": "unsupported_by_source",
                        "rationale": "The answer lacks source support.",
                    },
                ],
            }
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"action": "ignore"},
        {"schema_version": "oversight_arena.action.v2", "action": "accept_all"},
        {"action": "flag_errors", "unexpected": True},
        {
            "action": "flag_errors",
            "flags": [
                {
                    "answer_id": "",
                    "error_category": "numeric_mismatch",
                    "rationale": "The total does not match the source.",
                }
            ],
        },
        {
            "action": "flag_errors",
            "flags": [
                {
                    "answer_id": "answer-1",
                    "error_category": "not_a_category",
                    "rationale": "The total does not match the source.",
                }
            ],
        },
        {
            "action": "flag_errors",
            "flags": [
                {
                    "answer_id": "answer-1",
                    "error_category": "numeric_mismatch",
                    "rationale": "",
                }
            ],
        },
    ],
)
def test_action_rejects_bad_inputs(payload: dict[str, object]) -> None:
    """Malformed external action payloads fail validation."""

    with pytest.raises(ValidationError):
        OversightAction.model_validate(payload)


def test_public_contract_models_are_frozen() -> None:
    """Public schemas reject mutation of contract fields after validation."""

    observation = EpisodeManifest(
        episode_id="episode-1",
        source_records=(make_source_record(),),
        worker_answers=(make_worker_truth(),),
    ).to_observation()

    with pytest.raises(ValidationError, match="frozen"):
        observation.episode_id = "episode-2"
