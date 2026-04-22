---
name: python-env-hardening
description: Use when implementing or reviewing RL environment repositories, deterministic graders, parsers, training scripts, and test-heavy Python architecture.
---

When this skill is used:
- check for hidden-label leakage
- check deterministic behavior
- check separation of concerns
- check parser robustness
- check invalid action handling
- check seeded tests
- check that server layer is thin
- check consistency between prompt schema, parser schema, and action model

Always output:
1. risks found
2. fixes applied
3. tests added
4. remaining concerns