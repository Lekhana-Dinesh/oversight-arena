"""Repo-root wrapper for the Oversight Arena demo CLI."""

from __future__ import annotations

from pathlib import Path
import sys


def main() -> int:
    """Ensure the `src` package path is importable and run the package demo."""

    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from oversight_arena.demo import main as package_main

    return package_main()


if __name__ == "__main__":
    raise SystemExit(main())
