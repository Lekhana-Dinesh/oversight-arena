"""Tests for Oversight Arena prompt construction."""

from __future__ import annotations

from dataclasses import asdict
import json

import pytest

from oversight_arena.data_generator import Difficulty, Domain, generate_episode
from oversight_arena.environment import OversightArenaEnv
from oversight_arena.prompt_builder import AgentPrompt, PromptMessage, build_prompt


def make_observation() -> tuple[OversightArenaEnv, object]:
    """Return a deterministic one-answer environment observation."""

    env = OversightArenaEnv()
    observation = env.reset(
        seed=811,
        domain=Domain.FINANCE,
        difficulty=Difficulty.EASY,
        error_count=1,
    )
    return env, observation


def serialized_prompt(prompt: AgentPrompt) -> str:
    """Return prompt text as one searchable string."""

    return prompt.system + "\n" + prompt.user


def test_prompt_uses_only_public_observation_content() -> None:
    """Prompt text must not expose hidden truth or internal evidence fields."""

    env, observation = make_observation()
    prompt = build_prompt(observation)
    prompt_text = serialized_prompt(prompt)
    internal_payload = json.dumps(
        [asdict(truth) for truth in env.generated_episode().worker_truths],
        default=str,
    )

    assert isinstance(prompt, AgentPrompt)
    assert "is_correct" not in prompt_text
    assert "expected_answer" not in prompt_text
    assert "reviewer_note" not in prompt_text
    assert "evidence" not in prompt_text
    assert internal_payload not in prompt_text


def test_prompt_stably_includes_current_answer_and_source_data() -> None:
    """Prompt includes the current answer, all public sources, and action schema."""

    _env, observation = make_observation()
    current_answer = observation.worker_answers[0]
    prompt = build_prompt(observation)
    messages = prompt.as_messages()
    prompt_text = serialized_prompt(prompt)

    assert messages == (
        PromptMessage(role="system", content=prompt.system),
        PromptMessage(role="user", content=prompt.user),
    )
    assert current_answer.answer_id in prompt_text
    assert current_answer.question in prompt_text
    assert current_answer.answer in prompt_text
    assert all(record.record_id in prompt_text for record in observation.source_records)
    assert "accept_all" in prompt_text
    assert "flag_errors" in prompt_text
    assert "numeric_mismatch" in prompt_text


def test_prompt_builder_is_deterministic_for_same_observation() -> None:
    """Same public observation produces identical prompt pieces."""

    _env, observation = make_observation()

    assert build_prompt(observation) == build_prompt(observation)


def test_prompt_builder_rejects_multi_answer_observations() -> None:
    """Prompt builder supports the current one-answer-at-a-time contract only."""

    episode = generate_episode(seed=823, domain=Domain.RETAIL, difficulty=Difficulty.MEDIUM)

    with pytest.raises(ValueError, match="exactly one worker answer"):
        build_prompt(episode.to_observation())
