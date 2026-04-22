"""Core environment state machine for Oversight Arena.

This module owns reset/step transitions only. It wraps deterministic generated
episodes, exposes agent-visible observations, tracks mutable runtime state, and
delegates scoring to the pure grader module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from oversight_arena.data_generator import Difficulty, Domain, GeneratedEpisode, generate_episode
from oversight_arena.grader import AnswerGrade, EpisodeGrade, grade_episode
from oversight_arena.models import (
    ActionKind,
    FlaggedAnswer,
    OversightAction,
    OversightObservation,
    WorkerAnswer,
    WorkerAnswerTruth,
)


class EnvironmentError(Exception):
    """Base exception for invalid environment operations."""


class EnvironmentStateError(EnvironmentError):
    """Raised when reset/step ordering is invalid."""


class InvalidEnvironmentAction(EnvironmentError, ValueError):
    """Raised when an action is invalid for the current environment state."""


@dataclass(frozen=True, slots=True)
class StepResult:
    """Result of applying one action to the environment."""

    observation: OversightObservation | None
    done: bool
    reviewed_answer_id: str
    answer_grade: AnswerGrade | None
    final_grade: EpisodeGrade | None
    reviewed_count: int
    total_count: int


@dataclass(frozen=True, slots=True)
class _ReviewDecision:
    """Internal record of one reviewed answer decision."""

    answer_id: str
    flag: FlaggedAnswer | None


@dataclass(slots=True)
class _EpisodeState:
    """Mutable runtime state separated from public Pydantic observations."""

    episode: GeneratedEpisode
    current_index: int = 0
    decisions: dict[str, _ReviewDecision] = field(default_factory=dict)
    done: bool = False
    final_grade: EpisodeGrade | None = None

    @property
    def total_count(self) -> int:
        """Return number of answers in the generated episode."""

        return len(self.episode.worker_truths)

    @property
    def current_answer_id(self) -> str:
        """Return the answer ID currently available for review."""

        return self.episode.worker_truths[self.current_index].answer_id


class OversightArenaEnv:
    """Deterministic one-answer-at-a-time Oversight Arena environment."""

    def __init__(self) -> None:
        """Create an uninitialized environment."""

        self._state: _EpisodeState | None = None

    @property
    def done(self) -> bool:
        """Return whether the current episode has terminated."""

        state = self._require_state()
        return state.done

    @property
    def final_grade(self) -> EpisodeGrade | None:
        """Return the final episode grade after termination, if available."""

        state = self._require_state()
        return state.final_grade

    @property
    def reviewed_answer_ids(self) -> tuple[str, ...]:
        """Return answer IDs reviewed so far in deterministic review order."""

        state = self._require_state()
        return tuple(state.decisions)

    def reset(
        self,
        *,
        seed: int,
        domain: Domain | str = Domain.FINANCE,
        difficulty: Difficulty | str = Difficulty.EASY,
        error_count: int | None = None,
    ) -> OversightObservation:
        """Start a deterministic generated episode and return its first observation."""

        episode = generate_episode(
            seed=seed,
            domain=domain,
            difficulty=difficulty,
            error_count=error_count,
        )
        self._state = _EpisodeState(episode=episode)
        return self.current_observation()

    def current_observation(self) -> OversightObservation:
        """Return the current public observation without hidden truth metadata."""

        state = self._require_state()
        if state.done:
            raise EnvironmentStateError(
                "current_observation is unavailable after episode completion"
            )
        return _observation_for_current_answer(state)

    def step(self, action: OversightAction) -> StepResult:
        """Apply one review action and advance to the next answer or terminal state."""

        state = self._require_state()
        if state.done:
            raise EnvironmentStateError("cannot step after episode completion")

        current_answer_id = state.current_answer_id
        flag = _validated_current_flag(action, current_answer_id)
        if current_answer_id in state.decisions:
            raise InvalidEnvironmentAction(f"answer has already been reviewed: {current_answer_id}")

        current_truth = state.episode.worker_truths[state.current_index]
        answer_grade = _grade_single_decision(
            flag=flag,
            truth=current_truth,
            episode=state.episode,
        )
        state.decisions[current_answer_id] = _ReviewDecision(
            answer_id=current_answer_id,
            flag=flag,
        )
        state.current_index += 1

        if state.current_index >= state.total_count:
            final_grade = _grade_decisions(state.episode, state.decisions)
            state.done = True
            state.final_grade = final_grade
            return StepResult(
                observation=None,
                done=True,
                reviewed_answer_id=current_answer_id,
                answer_grade=answer_grade,
                final_grade=final_grade,
                reviewed_count=len(state.decisions),
                total_count=state.total_count,
            )

        return StepResult(
            observation=_observation_for_current_answer(state),
            done=False,
            reviewed_answer_id=current_answer_id,
            answer_grade=answer_grade,
            final_grade=None,
            reviewed_count=len(state.decisions),
            total_count=state.total_count,
        )

    def generated_episode(self) -> GeneratedEpisode:
        """Return the current generated episode for trusted internal callers."""

        return self._require_state().episode

    def _require_state(self) -> _EpisodeState:
        """Return initialized runtime state or raise a deterministic error."""

        if self._state is None:
            raise EnvironmentStateError("environment must be reset before use")
        return self._state


def _observation_for_current_answer(state: _EpisodeState) -> OversightObservation:
    """Build a public observation for exactly the current worker answer."""

    public_answer = state.episode.worker_truths[state.current_index].to_public_answer()
    return OversightObservation(
        episode_id=state.episode.episode_id,
        turn_index=state.current_index,
        source_records=state.episode.source_records,
        worker_answers=(public_answer,),
    )


def _validated_current_flag(
    action: OversightAction,
    current_answer_id: str,
) -> FlaggedAnswer | None:
    """Validate that an action applies to exactly the current answer."""

    flag_ids = [flag.answer_id for flag in action.flags]
    if len(set(flag_ids)) != len(flag_ids):
        raise InvalidEnvironmentAction("action contains duplicate flags")

    if action.action is ActionKind.ACCEPT_ALL:
        if action.flags:
            raise InvalidEnvironmentAction("accept_all actions must not include flags")
        return None

    if action.action is not ActionKind.FLAG_ERRORS:
        raise InvalidEnvironmentAction(f"unsupported action kind: {action.action}")

    if len(action.flags) != 1:
        raise InvalidEnvironmentAction("flag_errors must include exactly one flag per step")

    flag = action.flags[0]
    if flag.answer_id != current_answer_id:
        raise InvalidEnvironmentAction(
            f"action must review current answer_id {current_answer_id}, got {flag.answer_id}"
        )
    return flag


def _grade_decisions(
    episode: GeneratedEpisode,
    decisions: Mapping[str, _ReviewDecision],
) -> EpisodeGrade:
    """Aggregate one-answer decisions into a single grader action."""

    flags = tuple(decision.flag for decision in decisions.values() if decision.flag is not None)
    if flags:
        action = OversightAction(action=ActionKind.FLAG_ERRORS, flags=flags)
    else:
        action = OversightAction(action=ActionKind.ACCEPT_ALL)
    return grade_episode(action, episode.worker_truths, episode.source_records)


def _grade_single_decision(
    *,
    flag: FlaggedAnswer | None,
    truth: WorkerAnswerTruth,
    episode: GeneratedEpisode,
) -> AnswerGrade:
    """Grade one reviewed answer without affecting episode-level final scoring."""

    if flag is None:
        action = OversightAction(action=ActionKind.ACCEPT_ALL)
    else:
        action = OversightAction(action=ActionKind.FLAG_ERRORS, flags=(flag,))
    grade = grade_episode(action, (truth,), episode.source_records)
    return grade.answer_grades[0]
