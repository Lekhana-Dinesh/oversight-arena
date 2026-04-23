"""Thin FastAPI adapter for the Oversight Arena environment core.

This module keeps HTTP/session concerns out of the deterministic environment
core. Each reset creates a dedicated environment session, while the route layer
derives public metadata and schemas from the actual Pydantic models used by the
rest of the repository.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from oversight_arena import __version__
from oversight_arena.data_generator import Difficulty, Domain
from oversight_arena.environment import (
    EnvironmentStateError,
    InvalidEnvironmentAction,
    OversightArenaEnv,
    StepResult,
)
from oversight_arena.grader import EpisodeGrade
from oversight_arena.models import OversightAction, OversightObservation
from oversight_arena.server.session_store import (
    DEFAULT_SESSION_TTL_SECONDS,
    SessionNotFoundError,
    SessionRecord,
    SessionStore,
)


APP_TITLE = "Oversight Arena"
APP_DESCRIPTION = "Session-safe API adapter for the Oversight Arena environment core."


class HealthResponse(BaseModel):
    """Server health response compatible with the current OpenEnv validator."""

    status: Literal["healthy"] = "healthy"
    name: str = APP_TITLE
    version: str = __version__


class MetadataResponse(BaseModel):
    """Submission-facing environment metadata."""

    name: str
    description: str
    mode: Literal["simulation"] = "simulation"
    supported_domains: tuple[str, ...]
    supported_difficulties: tuple[str, ...]
    session_contract: str
    training_status: str


class SchemaResponse(BaseModel):
    """JSON schema report derived from the actual server and contract models."""

    action: dict[str, Any]
    observation: dict[str, Any]
    state: dict[str, Any]
    reset_request: dict[str, Any]
    reset_response: dict[str, Any]
    step_request: dict[str, Any]
    step_response: dict[str, Any]


class ResetRequest(BaseModel):
    """Request body for deterministic environment reset."""

    model_config = ConfigDict(extra="forbid")

    seed: int = Field(description="Deterministic episode seed.")
    domain: Domain = Field(default=Domain.FINANCE, description="Generated source-data domain.")
    difficulty: Difficulty = Field(
        default=Difficulty.EASY,
        description="Generated episode difficulty.",
    )
    error_count: int | None = Field(
        default=None,
        ge=0,
        description="Optional controlled number of injected worker errors.",
    )


class StepRequest(BaseModel):
    """Request body for applying one action to one active session."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(description="Opaque session handle returned by /reset.")
    action: OversightAction = Field(description="Public oversight action for the current turn.")


class ScoreResponse(BaseModel):
    """Public score-component summary safe for API responses."""

    precision_score: float
    recall_score: float
    reasoning_quality: float
    efficiency_score: float
    final_score: float

    @classmethod
    def from_grade(cls, grade: EpisodeGrade) -> "ScoreResponse":
        """Build a public score summary from an internal episode grade."""

        return cls(
            precision_score=grade.precision_score,
            recall_score=grade.recall_score,
            reasoning_quality=grade.reasoning_quality,
            efficiency_score=grade.efficiency_score,
            final_score=grade.final_score,
        )


class ResetResponse(BaseModel):
    """Response returned after starting a new session-scoped episode."""

    session_id: str
    expires_at: datetime
    observation: OversightObservation
    done: bool
    reviewed_count: int
    total_count: int
    final_grade: ScoreResponse | None = None


class StateResponse(BaseModel):
    """Current server-visible state for one session-scoped episode."""

    session_id: str
    expires_at: datetime
    observation: OversightObservation | None
    done: bool
    reviewed_count: int
    total_count: int
    final_grade: ScoreResponse | None


class StepResponse(BaseModel):
    """Response returned after applying one environment action."""

    session_id: str
    expires_at: datetime
    observation: OversightObservation | None
    done: bool
    reviewed_answer_id: str
    reviewed_count: int
    total_count: int
    final_grade: ScoreResponse | None


class McpError(BaseModel):
    """JSON-RPC error payload for the placeholder MCP endpoint."""

    code: int
    message: str


class McpResponse(BaseModel):
    """Minimal JSON-RPC response proving endpoint reachability."""

    jsonrpc: Literal["2.0"] = "2.0"
    id: Any | None = None
    error: McpError


def create_app(
    *,
    session_store: SessionStore | None = None,
) -> FastAPI:
    """Create a FastAPI app around per-session Oversight Arena environments."""

    app = FastAPI(
        title=APP_TITLE,
        version=__version__,
        description=APP_DESCRIPTION,
    )
    app.state.session_store = session_store or SessionStore(
        ttl_seconds=DEFAULT_SESSION_TTL_SECONDS
    )

    @app.exception_handler(EnvironmentStateError)
    async def handle_state_error(
        _request: Request,
        exc: EnvironmentStateError,
    ) -> JSONResponse:
        """Map invalid environment state transitions to HTTP 409."""

        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(InvalidEnvironmentAction)
    async def handle_invalid_action(
        _request: Request,
        exc: InvalidEnvironmentAction,
    ) -> JSONResponse:
        """Map semantically invalid environment actions to HTTP 400."""

        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(SessionNotFoundError)
    async def handle_missing_session(
        _request: Request,
        exc: SessionNotFoundError,
    ) -> JSONResponse:
        """Map unknown or expired session handles to HTTP 404."""

        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Return process-level server health."""

        _session_store(app).cleanup_expired_sessions()
        return HealthResponse()

    @app.get("/metadata", response_model=MetadataResponse)
    async def metadata() -> MetadataResponse:
        """Return a truthful summary of the current environment surface."""

        return MetadataResponse(
            name=APP_TITLE,
            description=(
                "Deterministic oversight-review environment with seeded synthetic data, "
                "one-answer-at-a-time transitions, and session-scoped HTTP episodes."
            ),
            supported_domains=tuple(domain.value for domain in Domain),
            supported_difficulties=tuple(difficulty.value for difficulty in Difficulty),
            session_contract=(
                "POST /reset returns session_id; GET /state and POST /step require it."
            ),
            training_status=(
                "Implements rollout collection and evaluation scaffolding, not a weight-update loop."
            ),
        )

    @app.get("/schema", response_model=SchemaResponse)
    async def schema() -> SchemaResponse:
        """Expose server and environment schemas derived from Pydantic models."""

        return SchemaResponse(
            action=OversightAction.model_json_schema(),
            observation=OversightObservation.model_json_schema(),
            state=StateResponse.model_json_schema(),
            reset_request=ResetRequest.model_json_schema(),
            reset_response=ResetResponse.model_json_schema(),
            step_request=StepRequest.model_json_schema(),
            step_response=StepResponse.model_json_schema(),
        )

    @app.post("/mcp", response_model=McpResponse)
    async def mcp(payload: dict[str, Any]) -> McpResponse:
        """Return a JSON-RPC error until a real MCP surface is implemented."""

        return McpResponse(
            id=payload.get("id"),
            error=McpError(
                code=-32601,
                message="MCP is not implemented for this repository yet.",
            ),
        )

    @app.post("/reset", response_model=ResetResponse)
    async def reset(request: ResetRequest) -> ResetResponse:
        """Reset the environment and return the first public observation."""

        store = _session_store(app)
        store.cleanup_expired_sessions()
        arena_env = OversightArenaEnv()
        try:
            observation = arena_env.reset(
                seed=request.seed,
                domain=request.domain,
                difficulty=request.difficulty,
                error_count=request.error_count,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        session = store.create(arena_env)
        return ResetResponse(
            session_id=session.session_id,
            expires_at=session.expires_at,
            observation=observation,
            done=False,
            reviewed_count=0,
            total_count=_total_count(session.env),
        )

    @app.get("/state", response_model=StateResponse)
    async def state(
        session_id: str = Query(..., description="Session handle returned by /reset."),
    ) -> StateResponse:
        """Return the current public observation or terminal score summary."""

        session = _session_store(app).get(session_id)
        return _state_response(session)

    @app.post("/step", response_model=StepResponse)
    async def step(request: StepRequest) -> StepResponse:
        """Apply one public oversight action to the environment core."""

        session = _session_store(app).get(request.session_id)
        result = session.env.step(request.action)
        return _step_response(result, session)

    return app


def _session_store(app: FastAPI) -> SessionStore:
    """Return the hosted session store."""

    store = app.state.session_store
    if not isinstance(store, SessionStore):
        raise RuntimeError("app.state.session_store must be a SessionStore")
    return store


def _state_response(session: SessionRecord) -> StateResponse:
    """Build a public current-state response from one session record."""

    env = session.env
    is_done = env.done
    final_grade = env.final_grade if is_done else None
    return StateResponse(
        session_id=session.session_id,
        expires_at=session.expires_at,
        observation=None if is_done else env.current_observation(),
        done=is_done,
        reviewed_count=len(env.reviewed_answer_ids),
        total_count=_total_count(env),
        final_grade=ScoreResponse.from_grade(final_grade) if final_grade is not None else None,
    )


def _step_response(result: StepResult, session: SessionRecord) -> StepResponse:
    """Build a public step response without exposing hidden labels."""

    return StepResponse(
        session_id=session.session_id,
        expires_at=session.expires_at,
        observation=result.observation,
        done=result.done,
        reviewed_answer_id=result.reviewed_answer_id,
        reviewed_count=result.reviewed_count,
        total_count=result.total_count,
        final_grade=(
            ScoreResponse.from_grade(result.final_grade)
            if result.final_grade is not None
            else None
        ),
    )


def _total_count(env: OversightArenaEnv) -> int:
    """Return the current episode answer count without exposing hidden truth."""

    return len(env.generated_episode().worker_truths)


app = create_app()

__all__ = [
    "APP_DESCRIPTION",
    "APP_TITLE",
    "HealthResponse",
    "McpError",
    "McpResponse",
    "MetadataResponse",
    "ResetRequest",
    "ResetResponse",
    "SchemaResponse",
    "ScoreResponse",
    "StateResponse",
    "StepRequest",
    "StepResponse",
    "app",
    "create_app",
]
