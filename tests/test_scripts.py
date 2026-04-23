"""Smoke tests for the repo-root demo and evaluation scripts."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_demo_script_runs_from_repo_root() -> None:
    """The baseline demo works from the repository root without extra path setup."""

    completed = subprocess.run(
        [sys.executable, "scripts/demo.py", "--seed", "1801", "--error-count", "0"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Oversight Arena Demo" in completed.stdout
    assert "Final grade:" in completed.stdout
    assert '"final_score": 1.0' in completed.stdout


def test_evaluate_script_runs_from_repo_root() -> None:
    """The baseline evaluation script emits a structured JSON report from repo root."""

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate.py",
            "--episodes",
            "1",
            "--seed-start",
            "1811",
            "--error-count",
            "0",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert '"name": "always_approve_baseline"' in completed.stdout
    assert '"average_final_score": 1.0' in completed.stdout
