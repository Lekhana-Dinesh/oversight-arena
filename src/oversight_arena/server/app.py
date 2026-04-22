"""Thin FastAPI adapter for the Oversight Arena environment core.

The server layer translates HTTP payloads into existing public action models
and delegates all reset, transition, generation, and grading behavior to
``OversightArenaEnv``. It intentionally exposes only public observations and
score summaries, never hidden truth records or evidence metadata.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from oversight_arena.data_generator import Difficulty, Domain
from oversight_arena.environment import (
    EnvironmentStateError,
    InvalidEnvironmentAction,
    OversightArenaEnv,
    StepResult,
)
from oversight_arena.grader import EpisodeGrade
from oversight_arena.models import OversightAction, OversightObservation


class HealthResponse(BaseModel):
    """Server health response."""

    status: str = "ok"


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
    """Response returned after starting a new episode."""

    observation: OversightObservation
    done: bool
    reviewed_count: int
    total_count: int
    final_grade: ScoreResponse | None = None


class StateResponse(BaseModel):
    """Current server-visible environment state."""

    observation: OversightObservation | None
    done: bool
    reviewed_count: int
    total_count: int
    final_grade: ScoreResponse | None


class StepResponse(BaseModel):
    """Response returned after applying one environment action."""

    observation: OversightObservation | None
    done: bool
    reviewed_answer_id: str
    reviewed_count: int
    total_count: int
    final_grade: ScoreResponse | None


def create_app(env: OversightArenaEnv | None = None) -> FastAPI:
    """Create a FastAPI app around one Oversight Arena environment instance."""

    app = FastAPI(
        title="Oversight Arena",
        version="0.1.0",
        description="Thin API adapter for the Oversight Arena environment core.",
    )
    app.state.environment = env or OversightArenaEnv()

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

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Return process-level server health."""

        return HealthResponse()

    @app.post("/reset", response_model=ResetResponse)
    async def reset(request: ResetRequest) -> ResetResponse:
        """Reset the environment and return the first public observation."""

        arena_env = _environment(app)
        try:
            observation = arena_env.reset(
                seed=request.seed,
                domain=request.domain,
                difficulty=request.difficulty,
                error_count=request.error_count,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return ResetResponse(
            observation=observation,
            done=False,
            reviewed_count=0,
            total_count=_total_count(arena_env),
        )

    @app.get("/state", response_model=StateResponse)
    async def state() -> StateResponse:
        """Return the current public observation or terminal score summary."""

        return _state_response(_environment(app))

    @app.post("/step", response_model=StepResponse)
    async def step(action: OversightAction) -> StepResponse:
        """Apply one public oversight action to the environment core."""

        result = _environment(app).step(action)
        return _step_response(result)

    return app


def _environment(app: FastAPI) -> OversightArenaEnv:
    """Return the hosted environment instance."""

    environment = app.state.environment
    if not isinstance(environment, OversightArenaEnv):
        raise RuntimeError("app.state.environment must be an OversightArenaEnv")
    return environment


def _state_response(env: OversightArenaEnv) -> StateResponse:
    """Build a public current-state response from the environment."""

    is_done = env.done
    final_grade = env.final_grade if is_done else None
    return StateResponse(
        observation=None if is_done else env.current_observation(),
        done=is_done,
        reviewed_count=len(env.reviewed_answer_ids),
        total_count=_total_count(env),
        final_grade=ScoreResponse.from_grade(final_grade) if final_grade is not None else None,
    )


def _step_response(result: StepResult) -> StepResponse:
    """Build a public step response without exposing per-answer hidden labels."""

    return StepResponse(
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
    "HealthResponse",
    "ResetRequest",
    "ResetResponse",
    "ScoreResponse",
    "StateResponse",
    "StepResponse",
    "app",
    "create_app",
]
