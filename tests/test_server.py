"""Tests for the thin Oversight Arena FastAPI adapter."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from oversight_arena.server.app import create_app


def make_client() -> TestClient:
    """Build an isolated test client with a fresh environment."""

    return TestClient(create_app())


def accept_action() -> dict[str, Any]:
    """Return a public accept-current-answer action payload."""

    return {"action": "accept_all"}


def flag_action(answer_id: str) -> dict[str, Any]:
    """Return a public flag-current-answer action payload."""

    return {
        "action": "flag_errors",
        "flags": [
            {
                "answer_id": answer_id,
                "error_category": "numeric_mismatch",
                "rationale": "finance-invoice-001 invoice_total",
            }
        ],
    }


def assert_no_hidden_truth(payload: dict[str, Any]) -> None:
    """Assert that API payloads do not expose hidden grading inputs."""

    serialized = json.dumps(payload)

    assert "is_correct" not in serialized
    assert "expected_answer" not in serialized
    assert "reviewer_note" not in serialized
    assert "evidence" not in serialized
    assert "expected_error_category" not in serialized
    assert "submitted_error_category" not in serialized
    assert "answer_grades" not in serialized


def reset_payload(
    *,
    seed: int = 701,
    domain: str = "finance",
    difficulty: str = "easy",
    error_count: int | None = None,
) -> dict[str, Any]:
    """Build a deterministic reset payload."""

    payload: dict[str, Any] = {
        "seed": seed,
        "domain": domain,
        "difficulty": difficulty,
    }
    if error_count is not None:
        payload["error_count"] = error_count
    return payload


def test_health_endpoint() -> None:
    """Health endpoint reports server readiness."""

    client = make_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_reset_endpoint_is_deterministic() -> None:
    """Reset through the API returns stable public observations."""

    client = make_client()
    payload = reset_payload(seed=703, domain="logistics", difficulty="hard", error_count=2)

    first = client.post("/reset", json=payload)
    second = client.post("/reset", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
    assert first.json()["done"] is False
    assert len(first.json()["observation"]["worker_answers"]) == 1
    assert first.json()["total_count"] == 5
    assert_no_hidden_truth(first.json())


def test_current_state_endpoint_before_and_after_reset() -> None:
    """Current-state endpoint exposes only initialized public state."""

    client = make_client()

    before_reset = client.get("/state")
    assert before_reset.status_code == 409
    assert "reset" in before_reset.json()["detail"]

    reset = client.post("/reset", json=reset_payload(seed=707, error_count=0))
    state = client.get("/state")

    assert reset.status_code == 200
    assert state.status_code == 200
    assert state.json()["observation"] == reset.json()["observation"]
    assert state.json()["done"] is False
    assert state.json()["final_grade"] is None
    assert_no_hidden_truth(state.json())


def test_valid_step_progression_and_terminal_grade() -> None:
    """Stepping through a zero-error episode reaches a terminal public score."""

    client = make_client()
    reset = client.post(
        "/reset",
        json=reset_payload(seed=709, domain="retail", difficulty="medium", error_count=0),
    )
    payload = reset.json()
    total_count = payload["total_count"]

    for expected_reviewed_count in range(1, total_count + 1):
        response = client.post("/step", json=accept_action())
        payload = response.json()

        assert response.status_code == 200
        assert payload["reviewed_count"] == expected_reviewed_count
        assert payload["total_count"] == total_count
        assert_no_hidden_truth(payload)

    assert payload["done"] is True
    assert payload["observation"] is None
    assert payload["final_grade"] == {
        "precision_score": pytest.approx(1.0),
        "recall_score": pytest.approx(1.0),
        "reasoning_quality": pytest.approx(1.0),
        "efficiency_score": pytest.approx(1.0),
        "final_score": pytest.approx(1.0),
    }

    terminal_state = client.get("/state")
    assert terminal_state.status_code == 200
    assert terminal_state.json()["done"] is True
    assert terminal_state.json()["observation"] is None
    assert terminal_state.json()["final_grade"]["final_score"] == pytest.approx(1.0)


def test_step_before_reset_and_after_done_are_api_errors() -> None:
    """Invalid environment ordering is surfaced as clear HTTP 409 errors."""

    client = make_client()

    before_reset = client.post("/step", json=accept_action())
    assert before_reset.status_code == 409
    assert "reset" in before_reset.json()["detail"]

    reset = client.post("/reset", json=reset_payload(seed=713, error_count=0))
    for _ in range(reset.json()["total_count"]):
        client.post("/step", json=accept_action())

    after_done = client.post("/step", json=accept_action())
    assert after_done.status_code == 409
    assert "after episode completion" in after_done.json()["detail"]


def test_invalid_action_errors_are_api_errors() -> None:
    """Environment-level action errors remain explicit through the API."""

    client = make_client()
    client.post("/reset", json=reset_payload(seed=719, domain="finance", difficulty="easy"))

    unknown_answer = client.post("/step", json=flag_action("missing-answer"))
    malformed_action = client.post("/step", json={"action": "flag_errors", "flags": []})

    assert unknown_answer.status_code == 400
    assert "current answer_id" in unknown_answer.json()["detail"]
    assert malformed_action.status_code == 422


def test_reset_validation_errors_are_api_errors() -> None:
    """Generator parameter validation is surfaced as an HTTP 400 response."""

    client = make_client()

    impossible_errors = client.post(
        "/reset",
        json=reset_payload(seed=727, difficulty="easy", error_count=4),
    )

    assert impossible_errors.status_code == 400
    assert "no greater" in impossible_errors.json()["detail"]


def test_nonterminal_step_response_does_not_expose_hidden_truth() -> None:
    """Server step responses keep observations public-only."""

    client = make_client()
    reset = client.post(
        "/reset",
        json=reset_payload(seed=733, domain="logistics", difficulty="medium", error_count=1),
    )
    current_answer_id = reset.json()["observation"]["worker_answers"][0]["answer_id"]

    response = client.post("/step", json=flag_action(current_answer_id))

    assert response.status_code == 200
    assert response.json()["done"] is False
    assert response.json()["final_grade"] is None
    assert_no_hidden_truth(response.json())
