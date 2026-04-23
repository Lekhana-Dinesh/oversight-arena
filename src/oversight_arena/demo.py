"""Command-line demo utilities for running one Oversight Arena episode."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from typing import Literal

from oversight_arena.adapters import openai_adapter_from_env
from oversight_arena.baseline import always_approve_response
from oversight_arena.data_generator import Difficulty, Domain
from oversight_arena.environment import OversightArenaEnv
from oversight_arena.inference import ModelOutputGenerator
from oversight_arena.parser import parse_action
from oversight_arena.prompt_builder import build_prompt


ProviderKind = Literal["baseline", "openai"]


@dataclass(frozen=True, slots=True)
class DemoConfig:
    """Configuration for a single human-readable demo episode."""

    seed: int
    domain: Domain
    difficulty: Difficulty
    error_count: int | None
    provider: ProviderKind
    model: str | None = None


def run_demo(config: DemoConfig) -> str:
    """Run one end-to-end episode and return a printable transcript."""

    env = OversightArenaEnv()
    generate_text = _build_generator(config)
    observation = env.reset(
        seed=config.seed,
        domain=config.domain,
        difficulty=config.difficulty,
        error_count=config.error_count,
    )

    lines = [
        "Oversight Arena Demo",
        f"episode_id: {observation.episode_id}",
        f"provider: {config.provider}",
        f"domain: {config.domain.value}",
        f"difficulty: {config.difficulty.value}",
        f"error_count: {config.error_count}",
    ]

    while True:
        prompt = build_prompt(observation)
        answer = observation.worker_answers[0]
        raw_output = generate_text(prompt.as_messages())
        parse_result = parse_action(raw_output)

        lines.extend(
            (
                "",
                f"Turn {observation.turn_index}",
                "Observation:",
                _pretty_json(observation.model_dump(mode="json")),
                "Raw model output:",
                raw_output,
                "Parse result:",
                _pretty_json(
                    {
                        "ok": parse_result.ok,
                        "error_type": (
                            parse_result.error_type.value
                            if parse_result.error_type is not None
                            else None
                        ),
                        "error_message": parse_result.error_message,
                        "action": (
                            parse_result.action.model_dump(mode="json")
                            if parse_result.action is not None
                            else None
                        ),
                    }
                ),
            )
        )

        if not parse_result.ok or parse_result.action is None:
            lines.extend(
                (
                    "Episode terminated:",
                    _pretty_json(
                        {
                            "terminal_reason": "invalid_parse",
                            "answer_id": answer.answer_id,
                        }
                    ),
                )
            )
            break

        step_result = env.step(parse_result.action)
        lines.extend(
            (
                "Answer grade:",
                _pretty_json(asdict(step_result.answer_grade) if step_result.answer_grade else None),
            )
        )

        if step_result.done:
            lines.extend(
                (
                    "Final grade:",
                    _pretty_json(
                        asdict(step_result.final_grade) if step_result.final_grade else None
                    ),
                )
            )
            break

        if step_result.observation is None:
            raise RuntimeError("non-terminal demo step did not return an observation")
        observation = step_result.observation

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the command-line demo and print its transcript."""

    parser = argparse.ArgumentParser(description="Run one Oversight Arena demo episode.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic episode seed.")
    parser.add_argument(
        "--domain",
        choices=[domain.value for domain in Domain],
        default=Domain.FINANCE.value,
        help="Synthetic source-data domain.",
    )
    parser.add_argument(
        "--difficulty",
        choices=[difficulty.value for difficulty in Difficulty],
        default=Difficulty.EASY.value,
        help="Episode difficulty level.",
    )
    parser.add_argument(
        "--error-count",
        type=int,
        default=None,
        help="Optional controlled number of injected worker errors.",
    )
    parser.add_argument(
        "--provider",
        choices=("baseline", "openai"),
        default="baseline",
        help="Text generator to use for the demo.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Provider model name for non-baseline demos.",
    )

    args = parser.parse_args(argv)
    if args.provider != "baseline" and (args.model is None or not args.model.strip()):
        parser.error("--model is required for non-baseline providers")

    config = DemoConfig(
        seed=args.seed,
        domain=Domain(args.domain),
        difficulty=Difficulty(args.difficulty),
        error_count=args.error_count,
        provider=args.provider,
        model=args.model,
    )
    print(run_demo(config))
    return 0


def _build_generator(config: DemoConfig) -> ModelOutputGenerator:
    """Return the configured generator callable for one demo run."""

    if config.provider == "baseline":
        return always_approve_response
    if config.provider == "openai":
        if config.model is None:
            raise RuntimeError("openai demos require a model name")
        return openai_adapter_from_env(model=config.model)
    raise RuntimeError(f"unsupported provider: {config.provider}")


def _pretty_json(payload: object) -> str:
    """Render structured payloads as readable deterministic JSON."""

    return json.dumps(_jsonable(payload), indent=2, sort_keys=True, default=str)


def _jsonable(payload: object) -> object:
    """Convert frozensets and nested structures into friendlier JSON values."""

    if isinstance(payload, dict):
        return {key: _jsonable(value) for key, value in payload.items()}
    if isinstance(payload, list | tuple):
        return [_jsonable(item) for item in payload]
    if isinstance(payload, set | frozenset):
        return sorted(_jsonable(item) for item in payload)
    return payload


__all__ = ["DemoConfig", "main", "run_demo"]
