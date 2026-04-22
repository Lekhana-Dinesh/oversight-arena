"""Training scaffold and metrics aggregation for Oversight Arena.

This module provides deterministic dry-run orchestration around the existing
rollout engine. It intentionally does not perform gradient updates, call model
provider SDKs, or own environment/grading logic; future training integrations
can use these typed summaries as their data-collection layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

from oversight_arena.baseline import always_approve_response
from oversight_arena.data_generator import Difficulty, Domain
from oversight_arena.inference import (
    ModelOutputGenerator,
    RolloutConfig,
    RolloutResult,
    TerminalReason,
    run_rollout,
)


class Curriculum(Protocol):
    """Callable protocol for deterministic rollout-parameter scheduling."""

    def __call__(self, episode_index: int, config: "TrainingConfig") -> RolloutConfig:
        """Return rollout parameters for one zero-based episode index."""


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Configuration for deterministic training-scaffold rollout collection."""

    episode_count: int
    seed_start: int = 0
    seed_stride: int = 1
    domains: tuple[Domain | str, ...] = (Domain.FINANCE,)
    difficulties: tuple[Difficulty | str, ...] = (Difficulty.EASY,)
    error_counts: tuple[int | None, ...] = (None,)
    run_name: str = "dry-run"

    def __post_init__(self) -> None:
        """Validate and normalize deterministic training configuration."""

        if self.episode_count <= 0:
            raise ValueError("episode_count must be positive")
        if self.seed_stride <= 0:
            raise ValueError("seed_stride must be positive")
        if not self.domains:
            raise ValueError("domains must include at least one domain")
        if not self.difficulties:
            raise ValueError("difficulties must include at least one difficulty")
        if not self.error_counts:
            raise ValueError("error_counts must include at least one value")
        if not self.run_name.strip():
            raise ValueError("run_name must not be blank")

        normalized_domains = tuple(Domain(domain) for domain in self.domains)
        normalized_difficulties = tuple(Difficulty(difficulty) for difficulty in self.difficulties)
        normalized_error_counts = tuple(self.error_counts)
        for error_count in normalized_error_counts:
            if error_count is not None and error_count < 0:
                raise ValueError("error_counts must not include negative values")

        object.__setattr__(self, "domains", normalized_domains)
        object.__setattr__(self, "difficulties", normalized_difficulties)
        object.__setattr__(self, "error_counts", normalized_error_counts)
        object.__setattr__(self, "run_name", self.run_name.strip())


@dataclass(frozen=True, slots=True)
class TerminalReasonMetric:
    """Count and rate for one rollout terminal reason."""

    terminal_reason: TerminalReason
    count: int
    rate: float


@dataclass(frozen=True, slots=True)
class TrainingMetrics:
    """Aggregated rollout metrics for a training-scaffold run."""

    episode_count: int
    completed_count: int
    invalid_parse_count: int
    invalid_action_count: int
    invalid_parse_rate: float
    invalid_action_rate: float
    average_final_score: float
    average_precision_score: float
    average_recall_score: float
    average_reasoning_quality: float
    average_efficiency_score: float
    average_steps_per_episode: float
    terminal_reason_distribution: tuple[TerminalReasonMetric, ...]

    def terminal_reason_counts(self) -> dict[str, int]:
        """Return terminal reason counts as a JSON-friendly dictionary."""

        return {
            metric.terminal_reason.value: metric.count
            for metric in self.terminal_reason_distribution
        }

    def to_report(self) -> dict[str, object]:
        """Return a JSON-serializable metrics report."""

        return {
            "episode_count": self.episode_count,
            "completed_count": self.completed_count,
            "invalid_parse_count": self.invalid_parse_count,
            "invalid_action_count": self.invalid_action_count,
            "invalid_parse_rate": self.invalid_parse_rate,
            "invalid_action_rate": self.invalid_action_rate,
            "average_final_score": self.average_final_score,
            "average_precision_score": self.average_precision_score,
            "average_recall_score": self.average_recall_score,
            "average_reasoning_quality": self.average_reasoning_quality,
            "average_efficiency_score": self.average_efficiency_score,
            "average_steps_per_episode": self.average_steps_per_episode,
            "terminal_reason_distribution": [
                {
                    "terminal_reason": metric.terminal_reason.value,
                    "count": metric.count,
                    "rate": metric.rate,
                }
                for metric in self.terminal_reason_distribution
            ],
        }


@dataclass(frozen=True, slots=True)
class TrainingRunResult:
    """Training-scaffold result containing rollouts, metrics, and report data."""

    config: TrainingConfig
    rollout_configs: tuple[RolloutConfig, ...]
    rollouts: tuple[RolloutResult, ...]
    metrics: TrainingMetrics

    def to_report(self) -> dict[str, object]:
        """Return a compact JSON-serializable training run report."""

        return {
            "run_name": self.config.run_name,
            "config": _config_report(self.config),
            "metrics": self.metrics.to_report(),
            "episodes": [
                {
                    "episode_id": rollout.episode_id,
                    "seed": rollout.config.seed,
                    "domain": rollout.config.domain.value,
                    "difficulty": rollout.config.difficulty.value,
                    "error_count": rollout.config.error_count,
                    "terminal_reason": rollout.terminal_reason.value,
                    "completed": rollout.completed,
                    "step_count": rollout.step_count,
                    "final_score": (
                        rollout.final_grade.final_score
                        if rollout.final_grade is not None
                        else None
                    ),
                }
                for rollout in self.rollouts
            ],
        }


def default_curriculum(episode_index: int, config: TrainingConfig) -> RolloutConfig:
    """Return deterministic rollout parameters for one scheduled episode."""

    return RolloutConfig(
        seed=config.seed_start + (episode_index * config.seed_stride),
        domain=config.domains[episode_index % len(config.domains)],
        difficulty=config.difficulties[episode_index % len(config.difficulties)],
        error_count=config.error_counts[episode_index % len(config.error_counts)],
    )


def run_training(
    config: TrainingConfig,
    *,
    generate_text: ModelOutputGenerator | None = None,
    curriculum: Curriculum | None = None,
) -> TrainingRunResult:
    """Collect rollout episodes and aggregate training-ready metrics.

    When ``generate_text`` is omitted, this function runs deterministic dry-run
    training with the always-approve baseline policy.
    """

    selected_generator = always_approve_response if generate_text is None else generate_text
    selected_curriculum = curriculum or default_curriculum

    rollout_configs: list[RolloutConfig] = []
    rollouts: list[RolloutResult] = []
    for episode_index in range(config.episode_count):
        rollout_config = selected_curriculum(episode_index, config)
        rollout_configs.append(rollout_config)
        rollouts.append(run_rollout(config=rollout_config, generate_text=selected_generator))

    rollout_tuple = tuple(rollouts)
    return TrainingRunResult(
        config=config,
        rollout_configs=tuple(rollout_configs),
        rollouts=rollout_tuple,
        metrics=aggregate_metrics(rollout_tuple),
    )


def aggregate_metrics(rollouts: Sequence[RolloutResult]) -> TrainingMetrics:
    """Aggregate deterministic metrics from rollout results.

    Score averages are computed over all scheduled rollouts. Episodes without a
    final grade contribute ``0.0`` to score averages and are separately counted
    in terminal-reason metrics.
    """

    if not rollouts:
        raise ValueError("rollouts must include at least one result")

    episode_count = len(rollouts)
    completed_count = sum(1 for rollout in rollouts if rollout.completed)
    invalid_parse_count = _terminal_count(rollouts, TerminalReason.INVALID_PARSE)
    invalid_action_count = _terminal_count(rollouts, TerminalReason.INVALID_ACTION)
    terminal_reason_distribution = tuple(
        TerminalReasonMetric(
            terminal_reason=terminal_reason,
            count=_terminal_count(rollouts, terminal_reason),
            rate=_rate(_terminal_count(rollouts, terminal_reason), episode_count),
        )
        for terminal_reason in TerminalReason
    )

    return TrainingMetrics(
        episode_count=episode_count,
        completed_count=completed_count,
        invalid_parse_count=invalid_parse_count,
        invalid_action_count=invalid_action_count,
        invalid_parse_rate=_rate(invalid_parse_count, episode_count),
        invalid_action_rate=_rate(invalid_action_count, episode_count),
        average_final_score=_mean(_final_score(rollout) for rollout in rollouts),
        average_precision_score=_mean(_precision_score(rollout) for rollout in rollouts),
        average_recall_score=_mean(_recall_score(rollout) for rollout in rollouts),
        average_reasoning_quality=_mean(_reasoning_quality(rollout) for rollout in rollouts),
        average_efficiency_score=_mean(_efficiency_score(rollout) for rollout in rollouts),
        average_steps_per_episode=_mean(float(rollout.step_count) for rollout in rollouts),
        terminal_reason_distribution=terminal_reason_distribution,
    )


def _terminal_count(
    rollouts: Sequence[RolloutResult],
    terminal_reason: TerminalReason,
) -> int:
    """Count rollouts ending with one terminal reason."""

    return sum(1 for rollout in rollouts if rollout.terminal_reason is terminal_reason)


def _rate(count: int, total: int) -> float:
    """Return a deterministic rate with zero-total protection."""

    if total == 0:
        return 0.0
    return count / total


def _mean(values: Iterable[float]) -> float:
    """Return the arithmetic mean for a finite iterable of floats."""

    value_tuple = tuple(values)
    if not value_tuple:
        return 0.0
    return sum(value_tuple) / len(value_tuple)


def _final_score(rollout: RolloutResult) -> float:
    """Return final score or zero for ungraded rollouts."""

    return 0.0 if rollout.final_grade is None else rollout.final_grade.final_score


def _precision_score(rollout: RolloutResult) -> float:
    """Return precision score or zero for ungraded rollouts."""

    return 0.0 if rollout.final_grade is None else rollout.final_grade.precision_score


def _recall_score(rollout: RolloutResult) -> float:
    """Return recall score or zero for ungraded rollouts."""

    return 0.0 if rollout.final_grade is None else rollout.final_grade.recall_score


def _reasoning_quality(rollout: RolloutResult) -> float:
    """Return reasoning score or zero for ungraded rollouts."""

    return 0.0 if rollout.final_grade is None else rollout.final_grade.reasoning_quality


def _efficiency_score(rollout: RolloutResult) -> float:
    """Return efficiency score or zero for ungraded rollouts."""

    return 0.0 if rollout.final_grade is None else rollout.final_grade.efficiency_score


def _config_report(config: TrainingConfig) -> dict[str, object]:
    """Return JSON-friendly training configuration metadata."""

    return {
        "episode_count": config.episode_count,
        "seed_start": config.seed_start,
        "seed_stride": config.seed_stride,
        "domains": [domain.value for domain in config.domains],
        "difficulties": [difficulty.value for difficulty in config.difficulties],
        "error_counts": list(config.error_counts),
    }


__all__ = [
    "Curriculum",
    "TerminalReasonMetric",
    "TrainingConfig",
    "TrainingMetrics",
    "TrainingRunResult",
    "aggregate_metrics",
    "default_curriculum",
    "run_training",
]
