"""Evaluation helpers for comparing Oversight Arena policies honestly."""

from __future__ import annotations

from dataclasses import dataclass

from oversight_arena.inference import ModelOutputGenerator
from oversight_arena.train import TrainingConfig, TrainingRunResult, run_training


@dataclass(frozen=True, slots=True)
class PolicyRunSummary:
    """Named evaluation result for one policy over a shared rollout schedule."""

    name: str
    result: TrainingRunResult

    def to_report(self) -> dict[str, object]:
        """Return a JSON-friendly report for one evaluated policy."""

        return {
            "name": self.name,
            "run": self.result.to_report(),
        }


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    """Side-by-side comparison of baseline and candidate policy evaluations."""

    baseline: PolicyRunSummary
    candidate: PolicyRunSummary
    metric_deltas: dict[str, float]

    def to_report(self) -> dict[str, object]:
        """Return a JSON-friendly comparison report."""

        return {
            "baseline": self.baseline.to_report(),
            "candidate": self.candidate.to_report(),
            "metric_deltas": self.metric_deltas,
        }


def evaluate_policy(
    *,
    name: str,
    config: TrainingConfig,
    generate_text: ModelOutputGenerator | None = None,
) -> PolicyRunSummary:
    """Run one policy over a shared schedule and return its structured summary."""

    return PolicyRunSummary(
        name=name,
        result=run_training(config, generate_text=generate_text),
    )


def compare_policies(
    *,
    config: TrainingConfig,
    candidate_name: str,
    candidate_generate_text: ModelOutputGenerator,
    baseline_name: str = "always_approve_baseline",
    baseline_generate_text: ModelOutputGenerator | None = None,
) -> ComparisonReport:
    """Evaluate a baseline and candidate policy over identical rollout schedules."""

    baseline = evaluate_policy(
        name=baseline_name,
        config=config,
        generate_text=baseline_generate_text,
    )
    candidate = evaluate_policy(
        name=candidate_name,
        config=config,
        generate_text=candidate_generate_text,
    )
    return ComparisonReport(
        baseline=baseline,
        candidate=candidate,
        metric_deltas=_metric_deltas(baseline.result, candidate.result),
    )


def _metric_deltas(
    baseline: TrainingRunResult,
    candidate: TrainingRunResult,
) -> dict[str, float]:
    """Return the most important candidate-minus-baseline metric deltas."""

    baseline_metrics = baseline.metrics
    candidate_metrics = candidate.metrics
    return {
        "average_final_score": (
            candidate_metrics.average_final_score - baseline_metrics.average_final_score
        ),
        "average_precision_score": (
            candidate_metrics.average_precision_score
            - baseline_metrics.average_precision_score
        ),
        "average_recall_score": (
            candidate_metrics.average_recall_score - baseline_metrics.average_recall_score
        ),
        "average_reasoning_quality": (
            candidate_metrics.average_reasoning_quality
            - baseline_metrics.average_reasoning_quality
        ),
        "average_efficiency_score": (
            candidate_metrics.average_efficiency_score
            - baseline_metrics.average_efficiency_score
        ),
        "invalid_parse_rate": (
            candidate_metrics.invalid_parse_rate - baseline_metrics.invalid_parse_rate
        ),
        "invalid_action_rate": (
            candidate_metrics.invalid_action_rate - baseline_metrics.invalid_action_rate
        ),
    }


__all__ = ["ComparisonReport", "PolicyRunSummary", "compare_policies", "evaluate_policy"]
