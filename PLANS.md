# Execution plan for Oversight Arena

## Goal
Implement a robust, reproducible, OpenEnv-compatible RL environment for oversight training.

## Deliverables
1. Strong typed schemas
2. Realistic deterministic task generation with seeded randomness
3. Deterministic grading
4. Environment transition logic with hidden metadata protection
5. OpenEnv-compatible server adapter
6. Agent prompt/response I/O
7. Baseline and inference scaffolding
8. Training orchestration scaffold
9. Full pytest coverage for critical paths

## Phases

### Phase 1: Contracts
- Implement models.py
- Define public observation/action schema
- Define internal worker answer / manifest schema
- Freeze action vocabulary and error taxonomy

### Phase 2: Grading
- Implement grader.py as pure logic
- Add deterministic precision/recall/reasoning/efficiency computation
- Add tests before environment integration

### Phase 3: Data generation
- Generate internally consistent examples across domains
- Attach question-answer pairs
- Inject controlled difficulty-specific errors
- Persist deterministic fixtures

### Phase 4: Environment
- Implement reset/step mechanics
- Ensure hidden metadata never enters agent-visible observations
- Enforce done semantics and invalid action handling

### Phase 5: Server adapter
- Build thin API/OpenEnv wrapper around environment core
- Avoid placing grading or transition logic in the web layer

### Phase 6: Agent I/O
- Build prompt builder and robust JSON parser
- Penalize invalid outputs cleanly and deterministically

### Phase 7: Baseline and inference
- Baseline rollout script
- Provider-agnostic inference scaffold
- Shared rollout result objects

### Phase 8: Training scaffold
- Repeated rollout collection
- Metrics aggregation
- Curriculum scheduling hooks
- Serializable dry-run reports

### Phase 9: Hardening
- Add edge-case tests
- Add smoke test
- Add contract validation
- Add README with exact run commands

## Review standard
Each phase is complete only if:
- code compiles
- relevant tests pass
- outputs are deterministic
- no hidden labels leak
- imports remain clean
