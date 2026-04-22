"""Tests for deterministic Oversight Arena baseline rollouts."""

from __future__ import annotations

import json

import pytest

from oversight_arena.baseline import always_approve_response, run_always_approve_baseline
from oversight_arena.data_generator import Difficulty, Domain
from oversight_arena.inference import TerminalReason
from oversight_arena.models import ActionKind
from oversight_arena.parser import parse_action


def test_always_approve_response_is_valid_public_action() -> None:
    """Baseline response parses into the public accept-all action."""

    result = parse_action(always_approve_response(()))

    assert result.ok
    assert result.action is not None
    assert result.action.action is ActionKind.ACCEPT_ALL
    assert result.action.flags == ()


def test_always_approve_baseline_is_deterministic() -> None:
    """Same baseline parameters produce identical rollout results."""

    first = run_always_approve_baseline(
        seed=901,
        domain=Domain.FINANCE,
        difficulty=Difficulty.MEDIUM,
        error_count=2,
    )
    second = run_always_approve_baseline(
        seed=901,
        domain=Domain.FINANCE,
        difficulty=Difficulty.MEDIUM,
        error_count=2,
    )

    assert first == second
    assert first.completed
    assert first.terminal_reason is TerminalReason.COMPLETED
    assert first.final_grade is not None
    assert first.step_count == first.final_grade.total_answers
    assert first.final_grade.total_flags == 0
    assert all(step.action is not None for step in first.steps)
    assert all(step.action.action is ActionKind.ACCEPT_ALL for step in first.steps if step.action)


def test_zero_error_always_approve_baseline_scores_perfectly() -> None:
    """Always approving is a perfect baseline only for zero-error episodes."""

    result = run_always_approve_baseline(
        seed=907,
        domain=Domain.RETAIL,
        difficulty=Difficulty.MEDIUM,
        error_count=0,
    )

    assert result.completed
    assert result.final_grade is not None
    assert result.final_grade.final_score == pytest.approx(1.0)
    assert all(step.answer_grade is not None for step in result.steps)


def test_baseline_rollout_prompts_do_not_leak_hidden_truth() -> None:
    """Baseline prompts and raw outputs remain public-action data only."""

    result = run_always_approve_baseline(
        seed=911,
        domain=Domain.LOGISTICS,
        difficulty=Difficulty.EASY,
        error_count=1,
    )
    serialized = json.dumps(
        [
            {
                "system": step.prompt.system,
                "user": step.prompt.user,
                "raw_output": step.raw_output,
                "action": step.action.model_dump(mode="json") if step.action else None,
            }
            for step in result.steps
        ]
    )

    assert "is_correct" not in serialized
    assert "expected_answer" not in serialized
    assert "reviewer_note" not in serialized
    assert "evidence" not in serialized
