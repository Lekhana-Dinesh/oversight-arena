# Oversight Arena

Oversight Arena is a Python project for building an OpenEnv-compatible reinforcement-learning environment where an oversight agent reviews worker-agent answers against structured source data. The agent's job is to decide whether the worker answers should be approved or flagged for specific errors.

The project is being implemented incrementally with a strong emphasis on stable public contracts, deterministic behavior, and strict separation between agent-visible observations and internal grading metadata.

## Current Status

Phase 1, contracts and schemas, is complete. Phase 2, deterministic grading, is complete, including the Phase 2.1 grading-semantics hardening pass. Phase 3, deterministic data generation, is complete. Phase 4, the core environment state machine, is complete. Phase 5, the thin server adapter, is complete. Phase 6, the agent I/O layer, is complete. Phase 7, baseline and inference scaffolding, is complete. Phase 8, training orchestration and metrics scaffolding, is complete.

Implemented:

- `pyproject.toml` with Python 3.11+ packaging metadata and minimal dependencies.
- `src/oversight_arena/models.py` with public observation/action schemas and internal truth schemas.
- `src/oversight_arena/grader.py` with deterministic episode scoring.
- `src/oversight_arena/data_generator.py` with seeded synthetic episode generation.
- `src/oversight_arena/environment.py` with deterministic reset/step transitions.
- `src/oversight_arena/server/app.py` with a thin FastAPI adapter around the environment core.
- `src/oversight_arena/prompt_builder.py` with deterministic public-observation prompt construction.
- `src/oversight_arena/parser.py` with deterministic JSON extraction and `OversightAction` validation.
- `src/oversight_arena/inference.py` with provider-agnostic rollout scaffolding.
- `src/oversight_arena/baseline.py` with a deterministic always-approve baseline.
- `src/oversight_arena/train.py` with deterministic rollout collection, curriculum hooks, and metric aggregation.
- Tests for contracts, grading, generation, environment transitions, server behavior, prompt construction, parser failure modes, baseline rollouts, inference rollouts, and training metrics.

Not yet implemented:

- Provider-specific model adapters, CLI scripts, and full production RL/fine-tuning pipelines.

## Architecture

The current implementation defines the contract boundary for later environment work.

### Public Schemas

Public schemas are Pydantic v2 models intended for OpenEnv-facing payloads:

- `SourceRecord`: structured source data visible to the agent.
- `WorkerAnswer`: worker output visible to the agent.
- `OversightObservation`: complete public observation for an episode turn.
- `FlaggedAnswer`: one answer flagged by the oversight agent.
- `OversightAction`: public action submitted by the agent.

Public models are frozen, reject unknown fields, and validate stable schema versions. These contracts should remain stable as the environment implementation grows.

### Internal Schemas

Internal schemas are plain frozen dataclasses:

- `WorkerAnswerTruth`: worker answer plus hidden correctness labels and expected answers.
- `EpisodeManifest`: internal episode manifest that projects into a public observation.

Hidden truth labels, expected answers, reviewer notes, and grading metadata are internal-only. They must never appear in agent-visible observations.

### Current Phase 2.1 Grading Semantics

The implemented grader is pure deterministic logic over an `OversightAction`, internal `WorkerAnswerTruth` records, and optional `SourceRecord` evidence.

- `precision_score`: true positive flags divided by total flags; when no flags are made, precision is `1.0` because there are no false positives, while missed errors are penalized through recall and reasoning.
- `recall_score`: true positive flags divided by total incorrect answers; when no incorrect answers exist, recall is `1.0`.
- `reasoning_quality`: true positive flags with matching category and cited supporting record or field divided by true positive flags; with no true positives, this is `1.0` only when there are zero incorrect answers and zero flags, otherwise `0.0`.
- `efficiency_score`: `1.0 - unnecessary_flags / total_answers`, clamped to `[0.0, 1.0]`; unnecessary flags are false positives plus category-mismatched true positives.
- `final_score`: weighted sum using precision `0.25`, recall `0.40`, reasoning `0.20`, and efficiency `0.15`.

Category mismatches still count as detecting an incorrect answer for precision and recall, but they reduce reasoning quality and efficiency.

### Current Phase 3 Data Generation

The implemented generator is pure in-memory logic for deterministic synthetic episodes. It supports the `finance`, `logistics`, and `retail` domains, with `easy`, `medium`, `hard`, and `expert` difficulty levels.

Generation is seed-aware: the same seed, domain, difficulty, and controlled error count produce the same source records, worker truths, and evidence metadata. Difficulty changes episode size, evidence complexity, and error category distribution.

Generated episodes keep public and internal data separated:

- Public projection: `SourceRecord`, `WorkerAnswer`, and `OversightObservation`.
- Internal projection: `WorkerAnswerTruth`, `EpisodeManifest`, and per-answer evidence metadata.

Each generated answer includes explicit internal evidence references to supporting source record IDs and field names. These references support the current grader's reasoning-citation checks and provide a future adapter point for environment prompts or stricter grading.

### Current Phase 4 Environment Core

The implemented environment core is a deterministic state machine over generated episodes. `OversightArenaEnv.reset(...)` starts a seeded episode and returns the first public observation; `step(...)` applies one `OversightAction` and returns a structured `StepResult`.

Observations are one-answer-at-a-time: each turn exposes all public source records plus exactly the current public `WorkerAnswer`. Mutable runtime state is held in an internal dataclass and kept separate from public Pydantic observation models.

Done semantics are stable: each valid step reviews one answer, the episode terminates after all generated answers are reviewed, and the terminal step returns `observation=None` plus the final `EpisodeGrade`. Final scoring is delegated to the grader rather than embedded in the environment.

Invalid transitions are explicit: stepping before reset, stepping after done, non-current answer IDs, duplicate flags, and malformed per-step flag counts raise environment-specific exceptions.

### Current Phase 5 Server Adapter

The implemented server adapter exposes health, reset, state, and step endpoints through FastAPI. It hosts an `OversightArenaEnv` instance and delegates generation, transitions, and grading to the core modules. Responses expose public observations and score summaries only; raw truth records, evidence metadata, and per-answer grading internals are not returned.

### Current Phase 6 Agent I/O

The implemented prompt builder converts the current one-answer `OversightObservation` into stable system/user prompt pieces using only public source records and the current public worker answer.

The parser accepts raw model text, extracts direct JSON, fenced JSON, or a bounded JSON object from surrounding prose, and validates it through `OversightAction`. Parse results distinguish successful parses from JSON extraction failures, malformed JSON, and schema validation failures. The parser does not call models, run inference loops, grade answers, or invent fallback actions.

### Current Phase 7 Baseline and Inference

The implemented rollout scaffolding runs complete episodes in-process by composing environment reset/step, prompt construction, injected model-output callables, response parsing, and structured rollout records. It does not hardcode any model provider SDK or perform network calls.

The baseline is a deterministic always-approve policy. Invalid model output follows a fail-fast policy: parse failures terminate the rollout with an explicit `invalid_parse` terminal reason and no fallback action is applied. Validly parsed actions that fail environment turn validation terminate with `invalid_action`.

### Current Phase 8 Training Scaffold

The implemented training scaffold orchestrates repeated in-process rollouts, supports deterministic dry-run collection with the baseline or an injected text generator, applies a configurable curriculum hook, and aggregates score and terminal-reason metrics.

This is not a full RL optimization pipeline: it performs no gradient updates, provider calls, checkpointing, or external experiment tracking. It is intended as a clean data-collection and metrics layer for later HF, TRL, Unsloth, or custom trainer integration.

## Repository Structure

```text
.
|-- AGENTS.md
|-- PLANS.md
|-- README.md
|-- pyproject.toml
|-- skills/
|   `-- python-env-hardening/
|       `-- SKILL.md
|-- src/
|   `-- oversight_arena/
|       |-- baseline.py
|       |-- data_generator.py
|       |-- environment.py
|       |-- grader.py
|       |-- inference.py
|       |-- models.py
|       |-- parser.py
|       |-- prompt_builder.py
|       |-- train.py
|       `-- server/
|           `-- app.py
`-- tests/
    |-- test_baseline.py
    |-- test_environment.py
    |-- test_generator.py
    |-- test_grader.py
    |-- test_inference.py
    |-- test_models.py
    |-- test_parser.py
    |-- test_prompt_builder.py
    |-- test_server.py
    `-- test_train.py
```

## Development Setup

Use Python 3.11 or newer.

```bash
python -m pip install -e ".[dev]"
```

Runtime dependencies are intentionally small: Pydantic v2 for external schema validation and FastAPI for the Phase 5 server adapter. The development extra includes pytest and HTTP client support for tests.

## Testing

Run the current Phase 1 through Phase 8 test suite:

```bash
python -m pytest tests/test_models.py tests/test_grader.py tests/test_generator.py tests/test_environment.py tests/test_server.py tests/test_prompt_builder.py tests/test_parser.py tests/test_baseline.py tests/test_inference.py tests/test_train.py
```

The tests cover public contract validation, invalid action inputs, internal truth consistency, immutable public models, hidden-label non-leakage, deterministic grading behavior, seeded generation, evidence metadata, environment reset/step transitions, done-state handling, server adapter behavior, prompt construction, parser validation, parser failure modes, deterministic baseline execution, provider-agnostic inference rollout behavior, curriculum scheduling, and training metric aggregation.

## Design Principles

- **Determinism**: grading, generation, and environment transitions should be reproducible for the same inputs and seeds.
- **Hidden-label non-leakage**: correctness labels, expected answers, and grading metadata remain internal-only.
- **Modularity**: schemas, grading, generation, parsing, environment transitions, and server integration are separate layers.
- **Typed Python**: production modules use explicit typing and small, focused objects.
- **Thin server layer**: the API adapter wraps environment logic without owning grading or transition behavior.

## Roadmap

Completed and planned phases:

1. **Contracts**: complete public observation/action schemas and internal truth schemas.
2. **Grader**: complete deterministic scoring logic for precision, recall, reasoning, and efficiency.
3. **Generator**: complete seeded task generation with structured source data, controlled worker errors, and explicit evidence metadata.
4. **Environment**: complete reset/step mechanics, one-answer-at-a-time observations, done-state handling, and invalid action behavior.
5. **Server adapter**: complete thin API wrapper around the environment core.
6. **Agent I/O**: complete deterministic prompt construction and JSON response parsing.
7. **Baseline and inference**: complete deterministic baseline and provider-agnostic rollout scaffolding.
8. **Training scaffold**: complete deterministic rollout collection, curriculum hooks, and metrics aggregation.
9. **Production training integrations**: planned provider-specific adapters, optimization loops, checkpointing, and experiment tracking.

Future components will be added phase by phase with tests for each new module.
