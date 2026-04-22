# Oversight Arena project instructions

## Objective
Build a production-quality Python project for an OpenEnv-compatible RL environment called Oversight Arena.
The environment trains an oversight agent to review worker-agent outputs against structured source data and flag incorrect answers.

## Non-negotiables
- Prioritize correctness, determinism, and clean architecture over speed.
- Never expose hidden labels or grading metadata to the agent observation.
- Keep domain logic separate from API/server code.
- Use typed Python throughout.
- Prefer pure functions for grading and data generation where practical.
- Add tests for every new module.
- Do not silently swallow exceptions.
- Do not introduce mock behavior into production modules unless explicitly isolated behind fixtures or test utilities.

## Architecture rules
- Source code lives under `src/oversight_arena/`.
- Tests live under `tests/`.
- Server adapter must be thin.
- Pydantic models may be used for external schemas and validation.
- Mutable episode runtime state should use plain Python dict/dataclass only where mutation safety is needed.
- The grader must be deterministic for the same inputs.

## Quality bar
- Each module must include docstrings.
- Use small functions with single responsibilities.
- Avoid giant classes.
- Avoid circular imports.
- Keep prompt-building, parsing, grading, and environment transitions as separate modules.

## Testing requirements
- Add unit tests for models, grader, generator, parser, and environment transitions.
- Add contract tests for invalid actions, done state, parser failures, and hidden-label non-leachage.
- Use seeded tests for reproducibility.

## Coding standards
- Python 3.11+
- Type hints everywhere
- No dead code
- No TODO placeholders in final implementation
- No print-debugging in library modules
- Use logging where needed
- Use Ruff-compatible style and pytest

## Workflow
Before making large edits:
1. Read `PLANS.md`
2. Propose the implementation plan
3. Implement one layer at a time
4. Run relevant tests
5. Report exactly what changed, what remains, and any risk