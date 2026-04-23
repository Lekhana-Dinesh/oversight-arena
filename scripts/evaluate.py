"""Repo-root wrapper for Oversight Arena policy evaluation and comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main(argv: list[str] | None = None) -> int:
    """Ensure the package path is importable and run evaluation from repo root."""

    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from oversight_arena.adapters import openai_adapter_from_env
    from oversight_arena.evaluation import compare_policies, evaluate_policy
    from oversight_arena.train import TrainingConfig

    parser = argparse.ArgumentParser(
        description="Evaluate the Oversight Arena baseline or compare it to a model adapter."
    )
    parser.add_argument("--episodes", type=int, default=3, help="Number of evaluation episodes.")
    parser.add_argument("--seed-start", type=int, default=100, help="Starting seed for evaluation.")
    parser.add_argument(
        "--domain",
        action="append",
        default=None,
        help="Domain value to include. Repeat to cycle multiple domains.",
    )
    parser.add_argument(
        "--difficulty",
        action="append",
        default=None,
        help="Difficulty value to include. Repeat to cycle multiple difficulties.",
    )
    parser.add_argument(
        "--error-count",
        action="append",
        dest="error_counts",
        type=int,
        default=None,
        help="Explicit error count to include. Repeat to cycle multiple values.",
    )
    parser.add_argument(
        "--provider",
        choices=("baseline", "openai"),
        default="baseline",
        help="Candidate provider to evaluate against the baseline.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Provider model name for non-baseline evaluations.",
    )
    args = parser.parse_args(argv)

    if args.provider != "baseline" and (args.model is None or not args.model.strip()):
        parser.error("--model is required for non-baseline providers")

    config = TrainingConfig(
        episode_count=args.episodes,
        seed_start=args.seed_start,
        domains=tuple(args.domain or ["finance"]),
        difficulties=tuple(args.difficulty or ["easy"]),
        error_counts=tuple(args.error_counts) if args.error_counts is not None else (None,),
        run_name="evaluation",
    )

    if args.provider == "baseline":
        report = evaluate_policy(name="always_approve_baseline", config=config).to_report()
    else:
        adapter = openai_adapter_from_env(model=args.model)
        report = compare_policies(
            config=config,
            candidate_name=f"openai:{args.model}",
            candidate_generate_text=adapter,
        ).to_report()

    import json

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
