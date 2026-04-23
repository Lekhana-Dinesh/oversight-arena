"""Tests for the optional OpenAI provider adapter."""

from __future__ import annotations

import pytest

from oversight_arena.adapters.openai_adapter import (
    OpenAIResponsesAdapter,
    openai_adapter_from_env,
)
from oversight_arena.prompt_builder import PromptMessage


class FakeResponsesAPI:
    """Record request arguments and return a controlled fake response."""

    def __init__(self, output_text: str | None = None) -> None:
        """Create a fake responses resource with one fixed response body."""

        self.output_text = output_text
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        """Store one create call and return a response-like object."""

        self.calls.append(kwargs)
        return type("FakeResponse", (), {"output_text": self.output_text})()


class FakeClient:
    """Expose the fake responses API under the expected client shape."""

    def __init__(self, output_text: str | None = None) -> None:
        """Create a fake client with one configurable output body."""

        self.responses = FakeResponsesAPI(output_text=output_text)


def test_openai_adapter_formats_messages_for_responses_api() -> None:
    """The adapter converts prompt messages into instructions plus input items."""

    client = FakeClient(output_text='{"action":"accept_all"}')
    adapter = OpenAIResponsesAdapter(model="gpt-test", client=client)

    output = adapter(
        (
            PromptMessage(role="system", content="System instructions."),
            PromptMessage(role="user", content="User asks for JSON."),
        )
    )

    assert output == '{"action":"accept_all"}'
    assert client.responses.calls == [
        {
            "model": "gpt-test",
            "instructions": "System instructions.",
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "User asks for JSON."}],
                }
            ],
        }
    ]


def test_openai_adapter_rejects_missing_output_text() -> None:
    """A provider response without text is surfaced as an explicit runtime error."""

    adapter = OpenAIResponsesAdapter(model="gpt-test", client=FakeClient(output_text=None))

    with pytest.raises(RuntimeError, match="output_text"):
        adapter((PromptMessage(role="user", content="Return JSON."),))


def test_openai_adapter_from_env_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment-based adapter construction fails clearly without credentials."""

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        openai_adapter_from_env(model="gpt-test")
