"""Tests for the Oversight Arena training scaffold."""

from __future__ import annotations

import json

import pytest

import oversight_arena.train as train_module
from oversight_arena.data_generator import Difficulty, Domain
from oversight_arena.inference import RolloutConfig, RolloutResult, TerminalReason
from oversight_arena.prompt_builder import PromptMessage
from oversight_arena.train import TrainingConfig, aggregate_metrics, default_curriculum, run_training


def accept_json(_messages: tuple[PromptMessage, ...]) -> str:
    """Return a valid accept-all model response."""

    return json.dumps({"action": "accept_all"})


def no_json(_messages: tuple[PromptMessage, ...]) -> str:
    """Return an invalid model response with no JSON action."""

    return "no valid action"


def wrong_answer_flag(_messages: tuple[PromptMessage, ...]) -> str:
    """Return a valid action schema that is invalid for current-turn semantics."""

    return json.dumps(
        {
            "action": "flag_errors",
            "flags": [
                {
                    "answer_id": "not-current",
                    "error_category": "numeric_mismatch",
                    "rationale": "finance-invoice-001 invoice_total",
                }
            ],
        }
    )


def test_deterministic_dry_run_training_with_baseline() -> None:
    """Default dry-run training uses the deterministic always-approve baseline."""

    config = TrainingConfig(
        episode_count=2,
        seed_start=1101,
        domains=("finance",),
        difficulties=("easy",),
        error_counts=(0,),
    )

    first = run_training(config)
    second = run_training(config)

    assert first == second
    assert first.metrics.episode_count == 2
    assert first.metrics.completed_count == 2
    assert first.metrics.average_final_score == pytest.approx(1.0)
    assert all(rollout.completed for rollout in first.rollouts)


def test_metric_aggregation_counts_invalid_rollouts_as_zero_scores() -> None:
    """Score averages are explicit over all scheduled episodes."""

    config = TrainingConfig(
        episode_count=2,
        seed_start=1111,
        domains=(Domain.FINANCE,),
        difficulties=(Difficulty.EASY,),
        error_counts=(0, 1),
    )

    result = run_training(config)

    assert result.metrics.average_final_score == pytest.approx(0.7)
    assert result.metrics.average_precision_score == pytest.approx(1.0)
    assert result.metrics.average_recall_score == pytest.approx(0.5)
    assert result.metrics.average_reasoning_quality == pytest.approx(0.5)
    assert result.metrics.average_efficiency_score == pytest.approx(1.0)
    assert result.metrics.average_steps_per_episode == pytest.approx(3.0)


def test_terminal_reason_distribution_and_invalid_parse_accounting() -> None:
    """Invalid parse rollouts are counted and rated deterministically."""

    result = run_training(
        TrainingConfig(episode_count=3, seed_start=1121, error_counts=(0,)),
        generate_text=no_json,
    )

    assert result.metrics.completed_count == 0
    assert result.metrics.invalid_parse_count == 3
    assert result.metrics.invalid_parse_rate == pytest.approx(1.0)
    assert result.metrics.invalid_action_count == 0
    assert result.metrics.terminal_reason_counts() == {
        "completed": 0,
        "invalid_parse": 3,
        "invalid_action": 0,
    }
    assert result.metrics.average_final_score == pytest.approx(0.0)


def test_invalid_action_accounting() -> None:
    """Valid JSON that fails environment semantics is tracked separately."""

    result = run_training(
        TrainingConfig(episode_count=2, seed_start=1131, error_counts=(1,)),
        generate_text=wrong_answer_flag,
    )

    assert result.metrics.completed_count == 0
    assert result.metrics.invalid_parse_count == 0
    assert result.metrics.invalid_action_count == 2
    assert result.metrics.invalid_action_rate == pytest.approx(1.0)
    assert all(rollout.terminal_reason is TerminalReason.INVALID_ACTION for rollout in result.rollouts)


def test_curriculum_hook_controls_rollout_parameters() -> None:
    """A curriculum callable can deterministically vary rollout parameters."""

    seen_indexes: list[int] = []

    def curriculum(index: int, config: TrainingConfig) -> RolloutConfig:
        seen_indexes.append(index)
        return RolloutConfig(
            seed=config.seed_start + index + 10,
            domain=Domain.RETAIL,
            difficulty=Difficulty.EASY,
            error_count=0,
        )

    result = run_training(
        TrainingConfig(episode_count=2, seed_start=1200),
        generate_text=accept_json,
        curriculum=curriculum,
    )

    assert seen_indexes == [0, 1]
    assert [rollout_config.seed for rollout_config in result.rollout_configs] == [1210, 1211]
    assert all(rollout_config.domain is Domain.RETAIL for rollout_config in result.rollout_configs)
    assert result.metrics.average_final_score == pytest.approx(1.0)


def test_default_curriculum_cycles_config_values() -> None:
    """Default curriculum cycles domains, difficulties, and error counts."""

    config = TrainingConfig(
        episode_count=3,
        seed_start=10,
        seed_stride=2,
        domains=("finance", "retail"),
        difficulties=("easy", "medium"),
        error_counts=(0, 1),
    )

    rollout_configs = tuple(default_curriculum(index, config) for index in range(3))

    assert rollout_configs == (
        RolloutConfig(seed=10, domain=Domain.FINANCE, difficulty=Difficulty.EASY, error_count=0),
        RolloutConfig(seed=12, domain=Domain.RETAIL, difficulty=Difficulty.MEDIUM, error_count=1),
        RolloutConfig(seed=14, domain=Domain.FINANCE, difficulty=Difficulty.EASY, error_count=0),
    )


def test_training_report_is_json_serializable_summary() -> None:
    """Training results expose compact artifact/logging report data."""

    result = run_training(
        TrainingConfig(
            episode_count=1,
            seed_start=1141,
            error_counts=(0,),
            run_name="unit-report",
        )
    )
    report = result.to_report()

    assert report["run_name"] == "unit-report"
    assert report["metrics"]["average_final_score"] == pytest.approx(1.0)
    assert report["episodes"][0]["terminal_reason"] == "completed"
    json.dumps(report)


def test_training_reuses_rollout_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Training orchestration calls the shared rollout function per episode."""

    calls: list[RolloutConfig] = []

    def fake_run_rollout(
        *,
        config: RolloutConfig,
        generate_text: object,
    ) -> RolloutResult:
        calls.append(config)
        return RolloutResult(
            config=config,
            episode_id=f"fake-{config.seed}",
            steps=(),
            completed=False,
            terminal_reason=TerminalReason.INVALID_PARSE,
            final_grade=None,
        )

    monkeypatch.setattr(train_module, "run_rollout", fake_run_rollout)

    result = run_training(
        TrainingConfig(episode_count=2, seed_start=1300),
        generate_text=accept_json,
    )

    assert tuple(calls) == result.rollout_configs
    assert result.metrics.invalid_parse_count == 2


def test_aggregate_metrics_rejects_empty_input() -> None:
    """Metric aggregation requires at least one rollout result."""

    with pytest.raises(ValueError, match="at least one"):
        aggregate_metrics(())
