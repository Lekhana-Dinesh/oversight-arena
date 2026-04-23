# Hackathon Submission Notes

## What This Environment Does

Oversight Arena is a deterministic environment for training or evaluating an oversight agent. The agent receives structured source records plus one worker answer at a time and must decide whether to accept the answer or flag a specific defect category.

This makes the environment useful for studying:

- oversight and review policies
- citation-aware error detection
- failure handling for invalid model outputs
- reproducible evaluation of approval vs flagging behavior

## Why It Matters

Many agent systems need a second-stage reviewer that can catch bad answers without seeing hidden labels. This repository provides a clean environment foundation for that setting:

- public observations stay separate from internal truth
- grading is deterministic and auditable
- runtime behavior is reproducible for fixed seeds
- the server surface is session-safe and externally runnable

## Implemented Today

- deterministic generator across finance, logistics, and retail domains
- deterministic episode grader with precision, recall, reasoning, and efficiency
- one-answer-at-a-time environment state machine
- session-safe FastAPI server with OpenEnv-aligned compatibility routes
- baseline no-credential demo path
- provider-agnostic rollout engine
- honest evaluation/comparison workflow
- optional OpenAI-backed adapter behind the existing generator protocol

## Still Scaffold-Only

- no gradient-based training loop
- no PPO/GRPO/DPO/SFT optimization pipeline
- no checkpoint management
- no experiment tracker integration
- no production MCP implementation

## How To Run The Demo

From the repository root:

```bash
python -m pip install -e ".[dev]"
python scripts/demo.py --seed 42 --difficulty easy --error-count 0
```

This demo requires no API key and prints:

- observation
- raw output
- parsed action
- answer grade
- final grade

## How To Run The Server

```bash
python -m server.app
```

Optional validation:

```bash
python -m openenv.cli validate .
python -m openenv.cli validate --url http://127.0.0.1:8000
```

## How To Run Tests

```bash
python -m pytest
```

## How The Model Adapter Plugs In

The rollout engine already accepts any callable that matches the existing prompt-message-to-text protocol. The included OpenAI adapter is a thin implementation of that protocol.

- adapter module: `src/oversight_arena/adapters/openai_adapter.py`
- required env var: `OPENAI_API_KEY`
- optional env var: `OPENAI_BASE_URL`

Evaluation example:

```bash
python scripts/evaluate.py --episodes 3 --provider openai --model <model-name>
```

## Known Limitations

- sessions are in-memory only
- `/mcp` is a reachability stub, not a real MCP server
- the OpenAI adapter is minimal and optional
- the training module is rollout/evaluation infrastructure, not a trainer

## Honest Pitch Claim

Use this exact claim or something equally narrow:

> Oversight Arena is a deterministic, session-safe environment foundation for oversight-agent training and evaluation. It includes seeded task generation, auditable grading, a real demo path, and a reproducible baseline-vs-model evaluation workflow. It is training-ready infrastructure, not yet a full RL optimization system.
