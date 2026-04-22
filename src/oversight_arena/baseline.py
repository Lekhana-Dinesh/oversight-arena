"""Deterministic baseline policies for Oversight Arena rollouts."""

from __future__ import annotations

import json

from oversight_arena.data_generator import Difficulty, Domain
from oversight_arena.environment import OversightArenaEnv
from oversight_arena.inference import RolloutResult, run_inference_episode
from oversight_arena.prompt_builder import PromptMessage


def always_approve_response(_messages: tuple[PromptMessage, ...]) -> str:
    """Return a deterministic accept-all model response for the current answer."""

    return json.dumps({"action": "accept_all"}, sort_keys=True)


def run_always_approve_baseline(
    *,
    seed: int,
    domain: Domain | str = Domain.FINANCE,
    difficulty: Difficulty | str = Difficulty.EASY,
    error_count: int | None = None,
    env: OversightArenaEnv | None = None,
) -> RolloutResult:
    """Run one episode with the deterministic always-approve baseline."""

    return run_inference_episode(
        seed=seed,
        domain=domain,
        difficulty=difficulty,
        error_count=error_count,
        generate_text=always_approve_response,
        env=env,
    )


__all__ = ["always_approve_response", "run_always_approve_baseline"]
