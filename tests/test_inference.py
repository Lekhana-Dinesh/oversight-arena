"""Tests for provider-agnostic Oversight Arena inference rollouts."""

from __future__ import annotations

import json

from oversight_arena.data_generator import Difficulty, Domain
from oversight_arena.inference import (
    RolloutConfig,
    TerminalReason,
    run_inference_episode,
    run_rollout,
)
from oversight_arena.models import ActionKind
from oversight_arena.parser import ParseErrorType
from oversight_arena.prompt_builder import PromptMessage


class QueuedModel:
    """Deterministic fake model that returns queued raw text outputs."""

    def __init__(self, outputs: tuple[str, ...]) -> None:
        """Create a fake model with fixed outputs."""

        self._outputs = outputs
        self.messages: list[tuple[PromptMessage, ...]] = []

    def __call__(self, messages: tuple[PromptMessage, ...]) -> str:
        """Return the next queued output and record prompt messages."""

        self.messages.append(messages)
        return self._outputs[len(self.messages) - 1]


def accept_json() -> str:
    """Return a valid accept-all action JSON payload."""

    return json.dumps({"action": "accept_all"})


def flag_wrong_answer_json() -> str:
    """Return a valid action schema that is invalid for the current turn."""

    return json.dumps(
        {
            "action": "flag_errors",
            "flags": [
                {
                    "answer_id": "not-current-answer",
                    "error_category": "numeric_mismatch",
                    "rationale": "finance-invoice-001 invoice_total",
                }
            ],
        }
    )


def test_valid_model_outputs_flow_through_full_episode() -> None:
    """Valid parsed outputs are applied through reset/step until terminal grade."""

    model = QueuedModel((accept_json(), accept_json(), accept_json()))

    result = run_inference_episode(
        seed=1001,
        domain=Domain.FINANCE,
        difficulty=Difficulty.EASY,
        error_count=0,
        generate_text=model,
    )

    assert result.completed
    assert result.terminal_reason is TerminalReason.COMPLETED
    assert result.final_grade is not None
    assert result.final_grade.final_score == 1.0
    assert result.step_count == 3
    assert len(model.messages) == 3
    assert all(step.parse_result.ok for step in result.steps)
    assert all(step.action is not None for step in result.steps)
    assert all(step.action.action is ActionKind.ACCEPT_ALL for step in result.steps if step.action)


def test_inference_builds_public_prompts_for_model_callable() -> None:
    """Injected model receives prompt messages built from public observations."""

    model = QueuedModel((accept_json(), accept_json(), accept_json()))

    result = run_inference_episode(
        seed=1003,
        domain=Domain.LOGISTICS,
        difficulty=Difficulty.EASY,
        error_count=0,
        generate_text=model,
    )
    serialized_messages = json.dumps(
        [
            {"role": message.role, "content": message.content}
            for messages in model.messages
            for message in messages
        ]
    )

    assert result.completed
    assert "Source records" in serialized_messages
    assert "Current worker answer" in serialized_messages
    assert "is_correct" not in serialized_messages
    assert "expected_answer" not in serialized_messages
    assert "reviewer_note" not in serialized_messages
    assert "evidence" not in serialized_messages


def test_invalid_json_terminates_without_applying_action() -> None:
    """No-JSON model output triggers the explicit invalid-parse policy."""

    model = QueuedModel(("I cannot decide.",))

    result = run_inference_episode(
        seed=1009,
        domain=Domain.RETAIL,
        difficulty=Difficulty.EASY,
        error_count=0,
        generate_text=model,
    )

    assert not result.completed
    assert result.terminal_reason is TerminalReason.INVALID_PARSE
    assert result.final_grade is None
    assert result.step_count == 1
    assert result.steps[0].action is None
    assert result.steps[0].answer_grade is None
    assert result.steps[0].parse_result.error_type is ParseErrorType.JSON_NOT_FOUND


def test_malformed_json_failure_is_propagated_in_rollout_result() -> None:
    """Malformed JSON is surfaced through the step parse result."""

    model = QueuedModel(('{"action": "accept_all"',))

    result = run_inference_episode(
        seed=1013,
        domain=Domain.FINANCE,
        difficulty=Difficulty.EASY,
        error_count=0,
        generate_text=model,
    )

    assert result.terminal_reason is TerminalReason.INVALID_PARSE
    assert result.steps[0].parse_result.error_type is ParseErrorType.MALFORMED_JSON
    assert result.steps[0].error_message == result.steps[0].parse_result.error_message


def test_validation_failure_is_propagated_in_rollout_result() -> None:
    """Parser validation failures stop rollout without a fallback action."""

    model = QueuedModel(('{"flags": []}',))

    result = run_inference_episode(
        seed=1017,
        domain=Domain.FINANCE,
        difficulty=Difficulty.EASY,
        error_count=0,
        generate_text=model,
    )

    assert result.terminal_reason is TerminalReason.INVALID_PARSE
    assert result.steps[0].parse_result.error_type is ParseErrorType.VALIDATION_ERROR
    assert result.steps[0].action is None
    assert result.final_grade is None


def test_semantically_invalid_action_terminates_without_final_grade() -> None:
    """Valid action JSON can still be rejected by environment turn semantics."""

    model = QueuedModel((flag_wrong_answer_json(),))

    result = run_inference_episode(
        seed=1021,
        domain=Domain.FINANCE,
        difficulty=Difficulty.EASY,
        error_count=1,
        generate_text=model,
    )

    assert not result.completed
    assert result.terminal_reason is TerminalReason.INVALID_ACTION
    assert result.final_grade is None
    assert result.steps[0].parse_result.ok
    assert result.steps[0].action is not None
    assert result.steps[0].answer_grade is None
    assert "current answer_id" in result.steps[0].error_message


def test_rollout_engine_is_deterministic_for_repeated_inputs() -> None:
    """Shared rollout engine is deterministic for identical config and model output."""

    config = RolloutConfig(
        seed=1027,
        domain=Domain.RETAIL,
        difficulty=Difficulty.EASY,
        error_count=0,
    )

    first = run_rollout(config=config, generate_text=lambda _messages: accept_json())
    second = run_rollout(config=config, generate_text=lambda _messages: accept_json())

    assert first == second
