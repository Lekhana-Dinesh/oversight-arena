"""OpenAI-backed model adapter for Oversight Arena rollouts."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, Protocol

from oversight_arena.prompt_builder import PromptMessage


class ResponsesAPI(Protocol):
    """Protocol for the OpenAI responses resource used by the adapter."""

    def create(self, **kwargs: Any) -> Any:
        """Create one model response from structured request arguments."""


class OpenAIClientLike(Protocol):
    """Protocol for the subset of the OpenAI client used by the adapter."""

    responses: ResponsesAPI


@dataclass(slots=True)
class OpenAIResponsesAdapter:
    """Callable adapter that satisfies the rollout engine's text-generator protocol."""

    model: str
    api_key: str | None = None
    base_url: str | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    timeout: float | None = None
    client: OpenAIClientLike | None = None
    _client: OpenAIClientLike = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate configuration and lazily build the provider client."""

        if not self.model.strip():
            raise ValueError("model must not be blank")
        self._client = self.client if self.client is not None else _build_client(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def __call__(self, messages: tuple[PromptMessage, ...]) -> str:
        """Generate raw model text for one Oversight Arena prompt turn."""

        if not messages:
            raise ValueError("messages must include at least one prompt message")

        instructions, input_items = _messages_to_responses_input(messages)
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
        }
        if instructions is not None:
            request_kwargs["instructions"] = instructions
        if self.max_output_tokens is not None:
            request_kwargs["max_output_tokens"] = self.max_output_tokens
        if self.temperature is not None:
            request_kwargs["temperature"] = self.temperature

        response = self._client.responses.create(**request_kwargs)
        output_text = getattr(response, "output_text", None)
        if not isinstance(output_text, str) or not output_text.strip():
            raise RuntimeError("OpenAI response did not include non-empty output_text")
        return output_text


def openai_adapter_from_env(
    *,
    model: str,
    api_key_env: str = "OPENAI_API_KEY",
    base_url_env: str = "OPENAI_BASE_URL",
    max_output_tokens: int | None = None,
    temperature: float | None = None,
    timeout: float | None = None,
) -> OpenAIResponsesAdapter:
    """Build an OpenAI adapter using environment variables for credentials."""

    api_key = os.getenv(api_key_env)
    if api_key is None or not api_key.strip():
        raise RuntimeError(f"missing required environment variable: {api_key_env}")

    base_url = os.getenv(base_url_env)
    return OpenAIResponsesAdapter(
        model=model,
        api_key=api_key,
        base_url=base_url.strip() if isinstance(base_url, str) and base_url.strip() else None,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        timeout=timeout,
    )


def _build_client(
    *,
    api_key: str | None,
    base_url: str | None,
    timeout: float | None,
) -> OpenAIClientLike:
    """Import and construct the OpenAI client only when the adapter is used."""

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI adapter requires the 'openai' package. Install the optional "
            "'openai' extra or provide a prebuilt client."
        ) from exc

    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def _messages_to_responses_input(
    messages: tuple[PromptMessage, ...],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert prompt-builder messages into OpenAI Responses API input items."""

    system_parts = [message.content for message in messages if message.role == "system"]
    input_items = [
        {
            "role": message.role,
            "content": [{"type": "input_text", "text": message.content}],
        }
        for message in messages
        if message.role != "system"
    ]
    instructions = "\n\n".join(system_parts) if system_parts else None
    return instructions, input_items


__all__ = ["OpenAIResponsesAdapter", "openai_adapter_from_env"]
