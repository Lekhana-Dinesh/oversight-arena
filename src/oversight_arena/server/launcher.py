"""Launch helpers for the Oversight Arena ASGI application."""

from __future__ import annotations

import uvicorn


def main() -> None:
    """Run the packaged Oversight Arena server with uvicorn."""

    uvicorn.run("server.app:app", host="127.0.0.1", port=8000, reload=False)


__all__ = ["main"]
