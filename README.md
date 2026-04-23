---
title: oversight-arena
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Oversight Arena

Oversight Arena is a deterministic reinforcement-learning environment for training and evaluating an oversight agent. The agent reviews worker answers against structured source records and decides whether to accept the answer or flag a specific error category.

The repository is built around a clean core:
- seeded synthetic task generation
- pure deterministic grading
- one-answer-at-a-time environment transitions
- hidden-label separation between public observations and internal truth
- provider-agnostic rollout scaffolding

It is now packaged as a session-safe, OpenEnv-aligned submission foundation with a runnable demo path, a thin optional provider adapter, and an honest evaluation workflow.

## What Is Implemented Today

- Stable public observation/action contracts in `src/oversight_arena/models.py`
- Deterministic grading in `src/oversight_arena/grader.py`
- Seeded synthetic episode generation in `src/oversight_arena/data_generator.py`
- Deterministic environment transitions in `src/oversight_arena/environment.py`
- Session-safe FastAPI server with OpenEnv-facing compatibility routes in `src/oversight_arena/server/app.py`
- Prompt construction and JSON parsing for model rollouts
- Deterministic baseline rollouts and provider-agnostic inference scaffolding
- Training-ready rollout collection and metrics aggregation in `src/oversight_arena/train.py`
- Evaluation helpers and repo-root demo/evaluation scripts
- An optional OpenAI-backed adapter behind the existing generator protocol

## What Is Not Implemented

- A real weight-update loop such as PPO, GRPO, DPO, or supervised fine-tuning
- Checkpointing, experiment tracking, or dataset export pipelines
- A production MCP implementation
- Provider-specific adapters beyond the included optional OpenAI example

## Quickstart

Run commands from the repository root: the directory that contains `pyproject.toml`, `openenv.yaml`, `src/`, and `tests/`.

Install the project:

```bash
python -m pip install -e ".[dev]"
```

Run the no-API-key baseline demo:

```bash
python scripts/demo.py --seed 42 --difficulty easy --error-count 0
```

Run the full test suite:

```bash
python -m pytest
```

Launch the HTTP server locally:

```bash
python -m server.app
```

If you use `uv`, the repo also supports:

```bash
uv run server
```

## OpenEnv Compatibility

This repository now includes:

- `openenv.yaml` at the project root
- a root `server/app.py` wrapper for OpenEnv-style launches
- `/health`, `/metadata`, `/schema`, `/mcp`, `/reset`, `/state`, and `/step`
- schemas derived from the actual Pydantic models rather than handwritten JSON
- `uv.lock` for reproducible `uv` workflows

Validate the local repository structure:

```bash
python -m openenv.cli validate .
```

Validate a running server:

```bash
python -m openenv.cli validate --url http://127.0.0.1:8000
```

### Session Contract

The HTTP server is session-scoped. Each call to `POST /reset` creates a new environment session and returns a `session_id`.

- `GET /state` requires `session_id` as a query parameter
- `POST /step` requires `session_id` plus the public `OversightAction`
- completed sessions remain readable until TTL eviction
- the default session TTL is 15 minutes and refreshes on access

Example `POST /step` body:

```json
{
  "session_id": "abc123...",
  "action": {
    "action": "accept_all"
  }
}
```

### OpenEnv-Facing Routes

- `GET /health`: validator-friendly process health
- `GET /metadata`: truthful environment metadata
- `GET /schema`: JSON schemas for action, observation, state, and transport models
- `POST /mcp`: JSON-RPC reachability stub; not a real MCP implementation
- `POST /reset`: create a fresh seeded episode session
- `GET /state`: inspect current or terminal state for one session
- `POST /step`: apply one action to one session

## Demo And Evaluation

### Baseline Demo

The default demo uses the deterministic always-approve baseline and requires no credentials:

```bash
python scripts/demo.py --seed 42 --domain finance --difficulty easy --error-count 0
```

The transcript prints:

- the current observation
- the raw model output
- the parsed action
- the per-answer grade
- the final episode grade

### Evaluation Workflow

The repository includes an honest evaluation path built on the existing rollout scaffold.

Evaluate the baseline only:

```bash
python scripts/evaluate.py --episodes 3 --seed-start 100 --error-count 0
```

Compare the baseline to a provider-backed policy:

```bash
python scripts/evaluate.py --episodes 3 --provider openai --model <model-name>
```

This produces structured JSON with:

- per-episode rollout outcomes
- aggregate metrics
- terminal-reason distributions
- candidate-minus-baseline metric deltas when comparing two policies

## Optional Provider Adapter

The included adapter lives in `src/oversight_arena/adapters/openai_adapter.py`.

To use it explicitly:

```bash
python -m pip install -e ".[dev,openai]"
```

Environment variables:

- `OPENAI_API_KEY`: required for the OpenAI adapter
- `OPENAI_BASE_URL`: optional for compatible proxy/base URLs

The adapter is intentionally thin:

- it satisfies the existing `ModelOutputGenerator` callable protocol
- it does not change inference logic
- it can be replaced by another adapter without touching `inference.py`

## Honest Training Status

`src/oversight_arena/train.py` is a rollout-collection and metrics module. It is useful infrastructure for future RL or fine-tuning work, but it is not itself a trainer.

What it does:

- schedules seeded rollout configurations
- runs repeated episodes
- aggregates scores and failure rates
- emits JSON-serializable reports

What it does not do:

- gradient updates
- checkpoint saving
- optimizer state management
- dataset export
- experiment tracking

The strongest honest claim is:

> this repository provides training-ready environment infrastructure plus a reproducible evaluation workflow, not a finished RL optimization stack.

## Architecture Summary

- `models.py`: public contracts and internal hidden-truth dataclasses
- `grader.py`: pure scoring logic
- `data_generator.py`: seeded synthetic task generation
- `environment.py`: deterministic reset/step state machine
- `server/`: thin HTTP/session adapter and launch helpers
- `prompt_builder.py`: public-observation prompt rendering
- `parser.py`: deterministic JSON extraction and action validation
- `baseline.py`: deterministic accept-all policy
- `inference.py`: rollout engine
- `train.py`: rollout scheduling and metrics aggregation
- `evaluation.py`: policy comparison helpers
- `adapters/`: optional provider integrations

## Testing

The test suite covers:

- schema validation and hidden-label non-leakage
- grading semantics
- seeded generation
- environment transitions and done-state behavior
- server behavior and session isolation
- prompt construction and parser failure modes
- baseline rollouts
- rollout/evaluation scaffolding
- repo-root demo and evaluation script smoke paths
- optional adapter wiring

Run everything with:

```bash
python -m pytest
```

## Known Limitations

- The included `/mcp` endpoint is a truthful placeholder, not a functional MCP server.
- The OpenAI adapter is optional and minimal by design; it does not add provider-specific retry or safety layers beyond the SDK defaults.
- Training remains scaffold-only until a real optimization loop is added.
- Sessions are in-memory only; they are not persisted across process restarts.

## Submission Note

For a hackathon submission, pair this README with `HACKATHON.md`. The README is developer-facing; `HACKATHON.md` is the concise reviewer-facing statement of scope, demo steps, and honest claims.
