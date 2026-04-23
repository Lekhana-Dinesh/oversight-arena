"""Tests for Oversight Arena policy evaluation helpers."""

from __future__ import annotations

import json

import pytest

from oversight_arena.evaluation import compare_policies, evaluate_policy
from oversight_arena.prompt_builder import PromptMessage
from oversight_arena.train import TrainingConfig


def accept_json(_messages: tuple[PromptMessage, ...]) -> str:
    """Return a valid accept-all action response."""

    return json.dumps({"action": "accept_all"})


def no_json(_messages: tuple[PromptMessage, ...]) -> str:
    """Return an invalid model response with no action JSON."""

    return "not valid JSON"


def test_evaluate_policy_reuses_training_scaffold_honestly() -> None:
    """Policy evaluation returns the existing rollout/metrics report rather than a fake trainer."""

    summary = evaluate_policy(
        name="baseline",
        config=TrainingConfig(episode_count=1, seed_start=1701, error_counts=(0,)),
    )

    assert summary.name == "baseline"
    assert summary.result.metrics.average_final_score == pytest.approx(1.0)
    assert summary.to_report()["run"]["metrics"]["average_final_score"] == pytest.approx(1.0)


def test_compare_policies_reports_candidate_minus_baseline_deltas() -> None:
    """Comparison reports surface candidate regressions explicitly."""

    report = compare_policies(
        config=TrainingConfig(episode_count=2, seed_start=1711, error_counts=(0,)),
        candidate_name="invalid-candidate",
        candidate_generate_text=no_json,
        baseline_generate_text=accept_json,
    )

    assert report.baseline.result.metrics.average_final_score == pytest.approx(1.0)
    assert report.candidate.result.metrics.invalid_parse_rate == pytest.approx(1.0)
    assert report.metric_deltas["average_final_score"] == pytest.approx(-1.0)
    assert report.metric_deltas["invalid_parse_rate"] == pytest.approx(1.0)
