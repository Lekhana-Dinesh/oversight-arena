"""Inference and rollout scaffolding for Oversight Arena episodes.

This module wires together environment reset/step transitions, prompt
construction, raw model-output parsing, and structured rollout records. Model
calling is deliberately abstract: callers inject a small callable that accepts
prompt messages and returns raw text.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from oversight_arena.data_generator import Difficulty, Domain
from oversight_arena.environment import InvalidEnvironmentAction, OversightArenaEnv
from oversight_arena.grader import AnswerGrade, EpisodeGrade
from oversight_arena.models import OversightAction
from oversight_arena.parser import ParseResult, parse_action
from oversight_arena.prompt_builder import AgentPrompt, PromptMessage, build_prompt


class ModelOutputGenerator(Protocol):
    """Callable protocol for provider-agnostic model output generation."""

    def __call__(self, messages: tuple[PromptMessage, ...]) -> str:
        """Return raw model text for one prompt."""


class TerminalReason(StrEnum):
    """Reasons an episode rollout stopped."""

    COMPLETED = "completed"
    INVALID_PARSE = "invalid_parse"
    INVALID_ACTION = "invalid_action"


@dataclass(frozen=True, slots=True)
class RolloutConfig:
    """Deterministic episode parameters for a rollout."""

    seed: int
    domain: Domain = Domain.FINANCE
    difficulty: Difficulty = Difficulty.EASY
    error_count: int | None = None


@dataclass(frozen=True, slots=True)
class RolloutStep:
    """Structured record for one rollout step."""

    turn_index: int
    answer_id: str
    prompt: AgentPrompt
    raw_output: str
    parse_result: ParseResult
    action: OversightAction | None
    answer_grade: AnswerGrade | None
    done: bool
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class RolloutResult:
    """Structured result for one complete or interrupted episode rollout."""

    config: RolloutConfig
    episode_id: str
    steps: tuple[RolloutStep, ...]
    completed: bool
    terminal_reason: TerminalReason
    final_grade: EpisodeGrade | None

    @property
    def step_count(self) -> int:
        """Return the number of recorded rollout steps."""

        return len(self.steps)


def run_inference_episode(
    *,
    seed: int,
    generate_text: ModelOutputGenerator,
    domain: Domain | str = Domain.FINANCE,
    difficulty: Difficulty | str = Difficulty.EASY,
    error_count: int | None = None,
    env: OversightArenaEnv | None = None,
) -> RolloutResult:
    """Run one episode using an injected model-output callable."""

    config = RolloutConfig(
        seed=seed,
        domain=Domain(domain),
        difficulty=Difficulty(difficulty),
        error_count=error_count,
    )
    return run_rollout(config=config, generate_text=generate_text, env=env)


def run_rollout(
    *,
    config: RolloutConfig,
    generate_text: ModelOutputGenerator,
    env: OversightArenaEnv | None = None,
) -> RolloutResult:
    """Run a deterministic episode rollout through the shared inference engine.

    Invalid parse policy: terminate immediately with ``INVALID_PARSE`` and do
    not invent or apply an action. Semantically invalid actions that pass parser
    validation but fail environment checks terminate with ``INVALID_ACTION``.
    """

    arena_env = env or OversightArenaEnv()
    observation = arena_env.reset(
        seed=config.seed,
        domain=config.domain,
        difficulty=config.difficulty,
        error_count=config.error_count,
    )
    episode_id = observation.episode_id
    steps: list[RolloutStep] = []

    while True:
        answer = observation.worker_answers[0]
        prompt = build_prompt(observation)
        raw_output = generate_text(prompt.as_messages())
        parse_result = parse_action(raw_output)

        if not parse_result.ok:
            steps.append(
                RolloutStep(
                    turn_index=observation.turn_index,
                    answer_id=answer.answer_id,
                    prompt=prompt,
                    raw_output=raw_output,
                    parse_result=parse_result,
                    action=None,
                    answer_grade=None,
                    done=True,
                    error_message=parse_result.error_message,
                )
            )
            return RolloutResult(
                config=config,
                episode_id=episode_id,
                steps=tuple(steps),
                completed=False,
                terminal_reason=TerminalReason.INVALID_PARSE,
                final_grade=None,
            )

        action = parse_result.action
        if action is None:
            raise RuntimeError("successful parse result must include an action")

        try:
            step_result = arena_env.step(action)
        except InvalidEnvironmentAction as exc:
            steps.append(
                RolloutStep(
                    turn_index=observation.turn_index,
                    answer_id=answer.answer_id,
                    prompt=prompt,
                    raw_output=raw_output,
                    parse_result=parse_result,
                    action=action,
                    answer_grade=None,
                    done=True,
                    error_message=str(exc),
                )
            )
            return RolloutResult(
                config=config,
                episode_id=episode_id,
                steps=tuple(steps),
                completed=False,
                terminal_reason=TerminalReason.INVALID_ACTION,
                final_grade=None,
            )

        steps.append(
            RolloutStep(
                turn_index=observation.turn_index,
                answer_id=answer.answer_id,
                prompt=prompt,
                raw_output=raw_output,
                parse_result=parse_result,
                action=action,
                answer_grade=step_result.answer_grade,
                done=step_result.done,
            )
        )

        if step_result.done:
            return RolloutResult(
                config=config,
                episode_id=episode_id,
                steps=tuple(steps),
                completed=True,
                terminal_reason=TerminalReason.COMPLETED,
                final_grade=step_result.final_grade,
            )

        if step_result.observation is None:
            raise RuntimeError("non-terminal environment step did not return an observation")
        observation = step_result.observation


__all__ = [
    "ModelOutputGenerator",
    "RolloutConfig",
    "RolloutResult",
    "RolloutStep",
    "TerminalReason",
    "run_inference_episode",
    "run_rollout",
]
