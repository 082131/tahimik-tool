# TAHIMIK — instructions for AI assistants

Read these two files before proposing any change:

1. **[CONTRIBUTING.md](CONTRIBUTING.md)** — branching, commits, PRs, tests,
   and the "Working with AI assistants" section, which is binding.
2. **[docs/SPEC-WORKFLOW.md](docs/SPEC-WORKFLOW.md)** — how features are
   specified before they are built, and how existing code is retrofitted
   with specs.

Short version, so nothing depends on those files being loaded:

- Never commit to `main`. Branch as `<type>/<short-description>`.
- Never invent an answer to a design question — ask the author.
- Hyperparameters live in `configs/`, never inline. Seed stays 42.
- No secrets, datasets, checkpoints, or `node_modules` in git.
- `python -m pytest tests/ -v` passes before anything merges.
- Features go through the spec loop. Interrogation first, then the spec,
  then the build — not the other way round.
- Keep the `Co-Authored-By:` trailer on assisted commits.

This is a thesis repository. The history is read by examiners, so the
reasoning behind a change matters as much as the change.
