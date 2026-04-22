"""Deterministic parsing of agent text into Oversight Arena actions.

The parser extracts a JSON object from raw model text and delegates final schema
validation to ``OversightAction``. It returns structured parse results instead
of raising for malformed model output, making it reusable from future inference
and training loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import json
import math
import re
from typing import Any, Final

from pydantic import ValidationError

from oversight_arena.models import OversightAction


_FENCED_JSON_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"```(?:json|JSON)?\s*(.*?)```",
    re.DOTALL,
)


class ParseErrorType(StrEnum):
    """Structured parser failure categories."""

    JSON_NOT_FOUND = "json_not_found"
    MALFORMED_JSON = "malformed_json"
    VALIDATION_ERROR = "validation_error"


@dataclass(frozen=True, slots=True)
class ParseResult:
    """Result of parsing raw model text into an oversight action."""

    ok: bool
    action: OversightAction | None
    error_type: ParseErrorType | None
    error_message: str | None
    raw_text: str
    extracted_json: str | None = None


class _PayloadValidationError(ValueError):
    """Internal error for parser-level payload checks before action validation."""


def parse_action(raw_text: str) -> ParseResult:
    """Parse raw model text into a validated ``OversightAction``.

    Extraction order is deterministic: direct JSON, fenced JSON, then balanced
    JSON-object snippets found inside surrounding prose. Missing or invalid
    critical fields are reported as validation failures rather than filled in.
    """

    if not raw_text.strip():
        return _failure(
            raw_text=raw_text,
            error_type=ParseErrorType.JSON_NOT_FOUND,
            error_message="no JSON object found in empty response",
        )

    candidates = _json_candidates(raw_text)
    if not candidates:
        error_type = (
            ParseErrorType.MALFORMED_JSON
            if "{" in raw_text or "}" in raw_text
            else ParseErrorType.JSON_NOT_FOUND
        )
        return _failure(
            raw_text=raw_text,
            error_type=error_type,
            error_message="no complete JSON object found",
        )

    malformed_errors: list[str] = []
    validation_errors: list[str] = []

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            malformed_errors.append(str(exc))
            continue

        try:
            action_payload = _action_payload(payload)
            action = OversightAction.model_validate(action_payload)
        except (_PayloadValidationError, ValidationError) as exc:
            validation_errors.append(str(exc))
            continue

        return ParseResult(
            ok=True,
            action=action,
            error_type=None,
            error_message=None,
            raw_text=raw_text,
            extracted_json=candidate,
        )

    if validation_errors:
        return _failure(
            raw_text=raw_text,
            error_type=ParseErrorType.VALIDATION_ERROR,
            error_message=validation_errors[0],
            extracted_json=candidates[0],
        )

    return _failure(
        raw_text=raw_text,
        error_type=ParseErrorType.MALFORMED_JSON,
        error_message=malformed_errors[0] if malformed_errors else "malformed JSON",
        extracted_json=candidates[0],
    )


def _failure(
    *,
    raw_text: str,
    error_type: ParseErrorType,
    error_message: str,
    extracted_json: str | None = None,
) -> ParseResult:
    """Build a deterministic failed parse result."""

    return ParseResult(
        ok=False,
        action=None,
        error_type=error_type,
        error_message=error_message,
        raw_text=raw_text,
        extracted_json=extracted_json,
    )


def _json_candidates(raw_text: str) -> tuple[str, ...]:
    """Return candidate JSON object strings in deterministic extraction order."""

    candidates: list[str] = []
    stripped = raw_text.strip()
    if stripped.startswith("{"):
        candidates.append(stripped)

    candidates.extend(match.group(1).strip() for match in _FENCED_JSON_PATTERN.finditer(raw_text))
    candidates.extend(_balanced_json_objects(raw_text))

    return _dedupe_non_empty(candidates)


def _balanced_json_objects(raw_text: str) -> tuple[str, ...]:
    """Extract balanced JSON-object-like snippets from arbitrary text."""

    objects: list[str] = []
    for index, character in enumerate(raw_text):
        if character != "{":
            continue
        end_index = _matching_object_end(raw_text, index)
        if end_index is not None:
            objects.append(raw_text[index : end_index + 1].strip())
    return tuple(objects)


def _matching_object_end(raw_text: str, start_index: int) -> int | None:
    """Return the matching closing brace for a JSON object start."""

    depth = 0
    in_string = False
    escaped = False

    for index in range(start_index, len(raw_text)):
        character = raw_text[index]

        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue

        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return index

    return None


def _dedupe_non_empty(candidates: list[str]) -> tuple[str, ...]:
    """Return non-empty candidates preserving first-seen order."""

    seen: set[str] = set()
    result: list[str] = []
    for candidate in candidates:
        normalized = candidate.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return tuple(result)


def _action_payload(payload: Any) -> dict[str, Any]:
    """Return a payload suitable for ``OversightAction`` validation."""

    if not isinstance(payload, dict):
        raise _PayloadValidationError("top-level JSON value must be an object")

    if "confidence" not in payload:
        return payload

    confidence = payload["confidence"]
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, int | float)
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        raise _PayloadValidationError("confidence must be a number in [0.0, 1.0]")

    return {key: value for key, value in payload.items() if key != "confidence"}


__all__ = ["ParseErrorType", "ParseResult", "parse_action"]
