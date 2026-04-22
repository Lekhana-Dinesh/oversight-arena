"""Tests for deterministic parsing of model text into oversight actions."""

from __future__ import annotations

import json

from oversight_arena.models import ActionKind, ErrorCategory
from oversight_arena.parser import ParseErrorType, ParseResult, parse_action


def flag_json() -> str:
    """Return a valid flag action JSON string."""

    return json.dumps(
        {
            "action": "flag_errors",
            "flags": [
                {
                    "answer_id": "answer-001",
                    "error_category": "numeric_mismatch",
                    "rationale": "finance-invoice-001 invoice_total",
                }
            ],
        }
    )


def assert_success(result: ParseResult) -> None:
    """Assert a successful parse result has a validated action."""

    assert result.ok
    assert result.action is not None
    assert result.error_type is None
    assert result.error_message is None
    assert result.extracted_json is not None


def assert_failure(result: ParseResult, error_type: ParseErrorType) -> None:
    """Assert a failed parse result is explicit and action-free."""

    assert not result.ok
    assert result.action is None
    assert result.error_type is error_type
    assert result.error_message


def test_parse_valid_json_action() -> None:
    """Direct JSON parses into the public action model."""

    result = parse_action('{"action": "accept_all"}')

    assert_success(result)
    assert result.action.action is ActionKind.ACCEPT_ALL
    assert result.action.flags == ()


def test_parse_fenced_json_action() -> None:
    """Markdown-fenced JSON is extracted deterministically."""

    result = parse_action(f"```json\n{flag_json()}\n```")

    assert_success(result)
    assert result.action.action is ActionKind.FLAG_ERRORS
    assert result.action.flags[0].answer_id == "answer-001"
    assert result.action.flags[0].error_category is ErrorCategory.NUMERIC_MISMATCH


def test_parse_json_surrounded_by_prose() -> None:
    """A balanced JSON object can be recovered from surrounding prose."""

    result = parse_action(f"I will flag it.\n{flag_json()}\nThat is my answer.")

    assert_success(result)
    assert result.action.action is ActionKind.FLAG_ERRORS


def test_malformed_json_returns_parse_failure_without_crashing() -> None:
    """Malformed JSON is reported without raising to callers."""

    result = parse_action('{"action": "accept_all"')

    assert_failure(result, ParseErrorType.MALFORMED_JSON)


def test_missing_required_fields_return_validation_failure() -> None:
    """Parser does not invent missing critical fields."""

    result = parse_action('{"flags": []}')

    assert_failure(result, ParseErrorType.VALIDATION_ERROR)


def test_invalid_enum_values_return_validation_failure() -> None:
    """Invalid action or error-category enums are schema validation failures."""

    result = parse_action(
        json.dumps(
            {
                "action": "flag_errors",
                "flags": [
                    {
                        "answer_id": "answer-001",
                        "error_category": "not_a_category",
                        "rationale": "finance-invoice-001 invoice_total",
                    }
                ],
            }
        )
    )

    assert_failure(result, ParseErrorType.VALIDATION_ERROR)


def test_invalid_confidence_returns_validation_failure() -> None:
    """Invalid optional confidence metadata is rejected explicitly."""

    result = parse_action('{"action": "accept_all", "confidence": 1.5}')

    assert_failure(result, ParseErrorType.VALIDATION_ERROR)
    assert "confidence" in result.error_message


def test_valid_confidence_metadata_is_not_part_of_action_output() -> None:
    """Valid confidence metadata can be ignored without entering action output."""

    result = parse_action('{"action": "accept_all", "confidence": 0.75}')

    assert_success(result)
    action_payload = result.action.model_dump(mode="json")
    assert "confidence" not in action_payload


def test_parser_is_deterministic_for_repeated_inputs() -> None:
    """Repeated parsing of identical text returns identical structured results."""

    raw_text = f"```json\n{flag_json()}\n```"

    assert parse_action(raw_text) == parse_action(raw_text)


def test_non_json_response_returns_json_not_found() -> None:
    """Responses without JSON produce a parse failure instead of a fallback."""

    result = parse_action("I think the answer is probably fine.")

    assert_failure(result, ParseErrorType.JSON_NOT_FOUND)


def test_parser_output_does_not_include_hidden_or_invented_fields() -> None:
    """Successful parser output is only the public action contract."""

    result = parse_action(flag_json())

    assert_success(result)
    serialized = json.dumps(result.action.model_dump(mode="json"))
    assert "is_correct" not in serialized
    assert "expected_answer" not in serialized
    assert "reviewer_note" not in serialized
    assert "confidence" not in serialized
