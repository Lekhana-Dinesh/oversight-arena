"""OpenEnv-facing root server wrapper for Oversight Arena."""

from __future__ import annotations

from oversight_arena.server.app import app
from oversight_arena.server.launcher import main as run_server


def main() -> None:
    """Run the packaged Oversight Arena server from the repository root."""

    run_server()


__all__ = ["app", "main"]

if __name__ == "__main__":
    main()
