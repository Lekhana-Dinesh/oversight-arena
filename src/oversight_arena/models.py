"""Stable contracts for Oversight Arena observations and actions.

This module intentionally contains only schema and validation code. Public
contracts use frozen Pydantic models so OpenEnv-facing payloads are explicit and
versioned, while hidden truth data is held in internal dataclasses that project
to public observations without leaking grading metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    model_validator,
)


ObservationSchemaVersion = Literal["oversight_arena.observation.v1"]
ActionSchemaVersion = Literal["oversight_arena.action.v1"]

OBSERVATION_SCHEMA_VERSION: ObservationSchemaVersion = "oversight_arena.observation.v1"
ACTION_SCHEMA_VERSION: ActionSchemaVersion = "oversight_arena.action.v1"

Identifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$",
    ),
]
FieldName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
PublicText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8192)]
RationaleText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class ContractModel(BaseModel):
    """Base class for immutable public Pydantic contract models."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ActionKind(StrEnum):
    """Stable public action vocabulary for an oversight review."""

    ACCEPT_ALL = "accept_all"
    FLAG_ERRORS = "flag_errors"


class ErrorCategory(StrEnum):
    """Stable taxonomy for worker-answer defects."""

    INCONSISTENT_WITH_SOURCE = "inconsistent_with_source"
    UNSUPPORTED_BY_SOURCE = "unsupported_by_source"
    MISSING_REQUIRED_DETAIL = "missing_required_detail"
    NUMERIC_MISMATCH = "numeric_mismatch"
    ENTITY_MISMATCH = "entity_mismatch"
    TEMPORAL_MISMATCH = "temporal_mismatch"
    INSTRUCTION_VIOLATION = "instruction_violation"


class SourceRecord(ContractModel):
    """Structured source data visible to the oversight agent."""

    record_id: Identifier = Field(description="Stable public identifier for this source record.")
    record_type: Identifier = Field(description="Stable public type for this source record.")
    fields: dict[FieldName, JsonValue] = Field(
        min_length=1,
        description="JSON-serializable structured fields available for review.",
    )


class WorkerAnswer(ContractModel):
    """Worker answer visible in an oversight observation."""

    answer_id: Identifier = Field(description="Stable public identifier for this worker answer.")
    question: PublicText = Field(description="Question that the worker attempted to answer.")
    answer: PublicText = Field(description="Worker-provided answer that the agent must review.")
    source_record_ids: tuple[Identifier, ...] = Field(
        min_length=1,
        description="Public source records that are relevant to this answer.",
    )

    @model_validator(mode="after")
    def require_unique_source_references(self) -> Self:
        """Reject duplicate source references for one worker answer."""

        if len(set(self.source_record_ids)) != len(self.source_record_ids):
            raise ValueError("source_record_ids must not contain duplicates")
        return self


class OversightObservation(ContractModel):
    """Public observation presented to an oversight agent."""

    schema_version: ObservationSchemaVersion = Field(
        default=OBSERVATION_SCHEMA_VERSION,
        description="Version of the public observation contract.",
    )
    episode_id: Identifier = Field(description="Stable public episode identifier.")
    turn_index: int = Field(default=0, ge=0, description="Zero-based environment turn index.")
    source_records: tuple[SourceRecord, ...] = Field(
        min_length=1,
        description="Structured source records available to the oversight agent.",
    )
    worker_answers: tuple[WorkerAnswer, ...] = Field(
        min_length=1,
        description="Worker answers to be reviewed against the source records.",
    )

    @model_validator(mode="after")
    def require_consistent_public_references(self) -> Self:
        """Ensure public IDs are unique and answer references target known records."""

        source_ids = [record.record_id for record in self.source_records]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("source_records must have unique record_id values")

        answer_ids = [answer.answer_id for answer in self.worker_answers]
        if len(set(answer_ids)) != len(answer_ids):
            raise ValueError("worker_answers must have unique answer_id values")

        known_source_ids = set(source_ids)
        unknown_refs = sorted(
            {
                source_id
                for answer in self.worker_answers
                for source_id in answer.source_record_ids
                if source_id not in known_source_ids
            }
        )
        if unknown_refs:
            joined_refs = ", ".join(unknown_refs)
            raise ValueError(f"worker answer references unknown source records: {joined_refs}")

        return self


class FlaggedAnswer(ContractModel):
    """One answer flagged by the oversight agent as erroneous."""

    answer_id: Identifier = Field(description="Public worker answer ID being flagged.")
    error_category: ErrorCategory = Field(description="Category of the suspected worker error.")
    rationale: RationaleText = Field(description="Brief public rationale for the flag.")


class OversightAction(ContractModel):
    """Public action submitted by an oversight agent."""

    schema_version: ActionSchemaVersion = Field(
        default=ACTION_SCHEMA_VERSION,
        description="Version of the public action contract.",
    )
    action: ActionKind = Field(description="Review action selected by the oversight agent.")
    flags: tuple[FlaggedAnswer, ...] = Field(
        default_factory=tuple,
        max_length=64,
        description="Flagged worker answers, required when action is flag_errors.",
    )

    @model_validator(mode="after")
    def require_action_consistency(self) -> Self:
        """Validate action vocabulary, flag presence, and duplicate flagged IDs."""

        if self.action is ActionKind.ACCEPT_ALL and self.flags:
            raise ValueError("accept_all actions must not include flags")
        if self.action is ActionKind.FLAG_ERRORS and not self.flags:
            raise ValueError("flag_errors actions must include at least one flag")

        flagged_ids = [flag.answer_id for flag in self.flags]
        if len(set(flagged_ids)) != len(flagged_ids):
            raise ValueError("flags must not contain duplicate answer_id values")

        return self


@dataclass(frozen=True, slots=True)
class WorkerAnswerTruth:
    """Internal worker-answer schema with hidden grading truth labels."""

    answer_id: str
    question: str
    answer: str
    source_record_ids: tuple[str, ...]
    is_correct: bool
    expected_answer: str | None = None
    error_category: ErrorCategory | None = None
    reviewer_note: str | None = None

    def __post_init__(self) -> None:
        """Validate hidden answer truth without exposing it to public schemas."""

        object.__setattr__(self, "answer_id", _require_identifier(self.answer_id, "answer_id"))
        object.__setattr__(self, "question", _require_text(self.question, "question"))
        object.__setattr__(self, "answer", _require_text(self.answer, "answer"))

        source_record_ids = tuple(
            _require_identifier(source_id, "source_record_ids") for source_id in self.source_record_ids
        )
        if not source_record_ids:
            raise ValueError("source_record_ids must include at least one source record")
        if len(set(source_record_ids)) != len(source_record_ids):
            raise ValueError("source_record_ids must not contain duplicates")
        object.__setattr__(self, "source_record_ids", source_record_ids)

        if self.is_correct:
            if self.error_category is not None:
                raise ValueError("correct worker answers must not declare an error_category")
            return

        if self.error_category is None:
            raise ValueError("incorrect worker answers must declare an error_category")
        if self.expected_answer is None or not self.expected_answer.strip():
            raise ValueError("incorrect worker answers must include a non-empty expected_answer")

        object.__setattr__(self, "expected_answer", self.expected_answer.strip())
        if self.reviewer_note is not None:
            object.__setattr__(self, "reviewer_note", _require_text(self.reviewer_note, "reviewer_note"))

    def to_public_answer(self) -> WorkerAnswer:
        """Project this hidden truth record into the public worker-answer contract."""

        return WorkerAnswer(
            answer_id=self.answer_id,
            question=self.question,
            answer=self.answer,
            source_record_ids=self.source_record_ids,
        )


@dataclass(frozen=True, slots=True)
class EpisodeManifest:
    """Internal immutable episode manifest with hidden answer truth."""

    episode_id: str
    source_records: tuple[SourceRecord, ...]
    worker_answers: tuple[WorkerAnswerTruth, ...]
    turn_index: int = 0

    def __post_init__(self) -> None:
        """Validate internal manifest consistency before projection."""

        object.__setattr__(self, "episode_id", _require_identifier(self.episode_id, "episode_id"))
        if self.turn_index < 0:
            raise ValueError("turn_index must be non-negative")

        source_records = tuple(self.source_records)
        if not source_records:
            raise ValueError("source_records must include at least one record")
        if not all(isinstance(record, SourceRecord) for record in source_records):
            raise TypeError("source_records must contain SourceRecord instances")

        worker_answers = tuple(self.worker_answers)
        if not worker_answers:
            raise ValueError("worker_answers must include at least one answer")
        if not all(isinstance(answer, WorkerAnswerTruth) for answer in worker_answers):
            raise TypeError("worker_answers must contain WorkerAnswerTruth instances")

        source_ids = [record.record_id for record in source_records]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("source_records must have unique record_id values")

        answer_ids = [answer.answer_id for answer in worker_answers]
        if len(set(answer_ids)) != len(answer_ids):
            raise ValueError("worker_answers must have unique answer_id values")

        known_source_ids = set(source_ids)
        unknown_refs = sorted(
            {
                source_id
                for answer in worker_answers
                for source_id in answer.source_record_ids
                if source_id not in known_source_ids
            }
        )
        if unknown_refs:
            joined_refs = ", ".join(unknown_refs)
            raise ValueError(f"worker answer truth references unknown source records: {joined_refs}")

        object.__setattr__(self, "source_records", source_records)
        object.__setattr__(self, "worker_answers", worker_answers)

    def to_observation(self) -> OversightObservation:
        """Project this internal manifest into a public, hidden-label-free observation."""

        return OversightObservation(
            episode_id=self.episode_id,
            turn_index=self.turn_index,
            source_records=self.source_records,
            worker_answers=tuple(answer.to_public_answer() for answer in self.worker_answers),
        )


def _require_identifier(value: str, field_name: str) -> str:
    """Validate and normalize an internal identifier."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    if len(normalized) > 128:
        raise ValueError(f"{field_name} must be at most 128 characters")
    if not normalized[0].isalnum():
        raise ValueError(f"{field_name} must start with an alphanumeric character")
    allowed_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.:-")
    if any(char not in allowed_chars for char in normalized):
        raise ValueError(f"{field_name} contains unsupported characters")
    return normalized


def _require_text(value: str, field_name: str) -> str:
    """Validate and normalize internal text."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    if len(normalized) > 8192:
        raise ValueError(f"{field_name} must be at most 8192 characters")
    return normalized
