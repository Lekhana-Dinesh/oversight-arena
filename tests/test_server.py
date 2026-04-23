"""Tests for the session-safe Oversight Arena FastAPI adapter."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from typing import Any

from fastapi.testclient import TestClient
import pytest

from oversight_arena.server.app import create_app
from oversight_arena.server.session_store import SessionStore


def make_client(session_store: SessionStore | None = None) -> TestClient:
    """Build an isolated test client with a configurable session store."""

    return TestClient(create_app(session_store=session_store))


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


def step_payload(session_id: str, action: dict[str, Any]) -> dict[str, Any]:
    """Wrap one public action with the required session handle."""

    return {"session_id": session_id, "action": action}


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


def comparable_reset_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove non-deterministic transport fields before comparison."""

    comparable = dict(payload)
    comparable.pop("session_id")
    comparable.pop("expires_at")
    return comparable


def test_openenv_facing_routes_are_present_and_truthful() -> None:
    """The compatibility routes expose health, metadata, schema, and MCP reachability."""

    client = make_client()

    health = client.get("/health")
    metadata = client.get("/metadata")
    schema = client.get("/schema")
    mcp = client.post("/mcp", json={"jsonrpc": "2.0", "id": 7, "method": "ping"})

    assert health.status_code == 200
    assert health.json()["status"] == "healthy"
    assert metadata.status_code == 200
    assert isinstance(metadata.json()["name"], str)
    assert isinstance(metadata.json()["description"], str)
    assert schema.status_code == 200
    assert isinstance(schema.json()["action"], dict)
    assert isinstance(schema.json()["observation"], dict)
    assert isinstance(schema.json()["state"], dict)
    assert "session_id" in schema.json()["step_request"]["properties"]
    assert mcp.status_code == 200
    assert mcp.json()["jsonrpc"] == "2.0"
    assert mcp.json()["id"] == 7
    assert mcp.json()["error"]["code"] == -32601


def test_reset_is_deterministic_except_for_session_transport_fields() -> None:
    """Reset through the API returns stable public observations with unique handles."""

    client = make_client()
    payload = reset_payload(seed=703, domain="logistics", difficulty="hard", error_count=2)

    first = client.post("/reset", json=payload)
    second = client.post("/reset", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["session_id"] != second.json()["session_id"]
    assert comparable_reset_payload(first.json()) == comparable_reset_payload(second.json())
    assert first.json()["done"] is False
    assert len(first.json()["observation"]["worker_answers"]) == 1
    assert first.json()["total_count"] == 5
    assert_no_hidden_truth(first.json())


def test_state_endpoint_requires_a_known_session_handle() -> None:
    """State requests fail clearly without a session or with an unknown one."""

    client = make_client()

    missing_handle = client.get("/state")
    unknown_handle = client.get("/state", params={"session_id": "missing"})
    reset = client.post("/reset", json=reset_payload(seed=707, error_count=0))
    state = client.get("/state", params={"session_id": reset.json()["session_id"]})

    assert missing_handle.status_code == 422
    assert unknown_handle.status_code == 404
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
    session_id = payload["session_id"]
    total_count = payload["total_count"]

    for expected_reviewed_count in range(1, total_count + 1):
        response = client.post("/step", json=step_payload(session_id, accept_action()))
        payload = response.json()

        assert response.status_code == 200
        assert payload["session_id"] == session_id
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

    terminal_state = client.get("/state", params={"session_id": session_id})
    assert terminal_state.status_code == 200
    assert terminal_state.json()["done"] is True
    assert terminal_state.json()["observation"] is None
    assert terminal_state.json()["final_grade"]["final_score"] == pytest.approx(1.0)


def test_interleaved_sessions_do_not_overwrite_each_other() -> None:
    """Separate reset calls create isolated environments for later state and step calls."""

    client = make_client()
    first = client.post("/reset", json=reset_payload(seed=711, domain="finance", difficulty="easy"))
    second = client.post("/reset", json=reset_payload(seed=712, domain="retail", difficulty="easy"))

    first_session_id = first.json()["session_id"]
    second_session_id = second.json()["session_id"]
    first_episode_id = first.json()["observation"]["episode_id"]
    second_episode_id = second.json()["observation"]["episode_id"]

    first_step = client.post("/step", json=step_payload(first_session_id, accept_action()))
    first_state = client.get("/state", params={"session_id": first_session_id})
    second_state = client.get("/state", params={"session_id": second_session_id})

    assert first_step.status_code == 200
    assert first_state.status_code == 200
    assert second_state.status_code == 200
    assert first_state.json()["observation"]["episode_id"] == first_episode_id
    assert second_state.json()["observation"]["episode_id"] == second_episode_id
    assert first_state.json()["observation"]["turn_index"] == 1
    assert second_state.json()["observation"]["turn_index"] == 0


def test_invalid_actions_and_reset_validation_are_api_errors() -> None:
    """Generator validation, unknown sessions, and bad actions remain explicit."""

    client = make_client()

    impossible_errors = client.post(
        "/reset",
        json=reset_payload(seed=727, difficulty="easy", error_count=4),
    )
    reset = client.post("/reset", json=reset_payload(seed=719, domain="finance", difficulty="easy"))
    current_answer_id = reset.json()["observation"]["worker_answers"][0]["answer_id"]
    unknown_session = client.post("/step", json=step_payload("missing-session", accept_action()))
    unknown_answer = client.post(
        "/step",
        json=step_payload(reset.json()["session_id"], flag_action("missing-answer")),
    )
    malformed_action = client.post(
        "/step",
        json={"session_id": reset.json()["session_id"], "action": {"action": "flag_errors", "flags": []}},
    )
    valid_flag = client.post(
        "/step",
        json=step_payload(reset.json()["session_id"], flag_action(current_answer_id)),
    )

    assert impossible_errors.status_code == 400
    assert "no greater" in impossible_errors.json()["detail"]
    assert unknown_session.status_code == 404
    assert unknown_answer.status_code == 400
    assert "current answer_id" in unknown_answer.json()["detail"]
    assert malformed_action.status_code == 422
    assert valid_flag.status_code == 200


def test_step_response_does_not_expose_hidden_truth() -> None:
    """Server step responses keep observations public-only."""

    client = make_client()
    reset = client.post(
        "/reset",
        json=reset_payload(seed=733, domain="logistics", difficulty="medium", error_count=1),
    )
    current_answer_id = reset.json()["observation"]["worker_answers"][0]["answer_id"]

    response = client.post(
        "/step",
        json=step_payload(reset.json()["session_id"], flag_action(current_answer_id)),
    )

    assert response.status_code == 200
    assert response.json()["done"] is False
    assert response.json()["final_grade"] is None
    assert_no_hidden_truth(response.json())


def test_session_ttl_expiry_evicts_completed_or_idle_sessions() -> None:
    """Expired sessions are removed from the adapter without touching core logic."""

    current_time = datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc)

    def fake_now() -> datetime:
        return current_time

    store = SessionStore(ttl_seconds=30, now=fake_now)
    client = make_client(session_store=store)

    reset = client.post("/reset", json=reset_payload(seed=739, error_count=0))
    session_id = reset.json()["session_id"]

    current_time = current_time + timedelta(seconds=31)
    expired_state = client.get("/state", params={"session_id": session_id})

    assert expired_state.status_code == 404
    assert "session_id" in expired_state.json()["detail"]
