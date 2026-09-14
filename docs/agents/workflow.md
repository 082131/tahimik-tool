# TAHIMIK agent workflow

Before proposing a change, read:

1. [contributing.md](contributing.md) for repository rules, tests, review, and
   commit requirements.
2. [../SPEC-WORKFLOW.md](../SPEC-WORKFLOW.md) for the required Spec Kit flow.
3. [../../specs/FINDINGS.md](../../specs/FINDINGS.md) for recorded
   manuscript-to-code gaps.

Core rules:

- Do not invent answers to open design questions. Ask the author or record
  `NEEDS CLARIFICATION`.
- Put hyperparameters in `configs/`; retain seed 42 unless an approved
  experiment changes it.
- Do not commit secrets, datasets, checkpoints, dependency trees, or `tmp/`.
- Run `python -m pytest tests/ -v` before proposing a merge.
- Preserve the required `Co-Authored-By:` trailer on assisted commits.
