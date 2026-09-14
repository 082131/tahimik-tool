# Workspace Organization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Relocate local thesis material and durable documentation into predictable folders while preserving project-tool entry points.

**Architecture:** `tmp/` is the ignored home for local-only and regenerable artifacts. `docs/agents/` holds the substantive guidance an AI agent must follow, reached by a minimal root-level `CLAUDE.md` discovery pointer. `docs/project/` holds durable project records.

**Tech Stack:** Git, PowerShell, pytest configuration, Markdown documentation.

**Spec:** User-approved workspace map in the 2026-09-14 Codex conversation.

## Global Constraints

- Keep `README.md`, `requirements.txt`, `pytest.ini`, `conftest.py`, and `.gitignore` at the repository root because developer tools discover them there.
- Keep a minimal `CLAUDE.md` at the root solely as a pointer to `docs/agents/README.md`.
- Preserve the contents and history of every moved file with `git mv`.
- Treat `tmp/` as ignored local-only storage; retain only a `.gitkeep` marker in Git.

---

### Task 1: Establish the organized documentation homes

**Files:**
- Create: `docs/agents/README.md`
- Move: `CLAUDE.md` to `docs/agents/workflow.md`
- Move: `CONTRIBUTING.md` to `docs/agents/contributing.md`
- Move: `ASSESSMENT.md` to `docs/project/assessment.md`
- Move: `ATTRIBUTIONS.md` to `docs/project/attributions.md`
- Move: `HANDOFF.md` to `docs/project/handoff.md`
- Modify: root `CLAUDE.md`

- [ ] Move durable instruction and project-record documents into their categorized folders.
- [ ] Create an agent-document index that states the required reading order.
- [ ] Replace the root agent instruction file with a one-purpose discovery pointer.
- [ ] Verify every moved document exists and the root pointer resolves.

### Task 2: Relocate and exclude local thesis material

**Files:**
- Move: `thesis_code/` to `tmp/thesis/`
- Modify: `.gitignore`
- Create: `tmp/.gitkeep`

- [ ] Move the local thesis source tree to `tmp/thesis/`.
- [ ] Replace the obsolete `thesis_code/` ignore rule with scoped `tmp/` exclusions, allowing only its directory marker to remain tracked.
- [ ] Verify Git sees no thesis source files as candidates to commit.

### Task 3: Repair internal documentation and source references

**Files:**
- Modify: repository Markdown and Python files that reference the old paths.
- Modify: `README.md` repository-layout and documentation links.
- Modify: `pytest.ini` exclusion path.

- [ ] Update exact file-path references to the new documentation and thesis locations.
- [ ] Preserve links to the root `CLAUDE.md` when they intentionally describe the automatic discovery entry point.
- [ ] Verify that no operational reference remains to `thesis_code/`, root project-record Markdown, or root `CONTRIBUTING.md`.

### Task 4: Verify the new organization

**Files:**
- Test: repository path and reference checks

- [ ] Run `git status --short` to distinguish intentional renames from pre-existing worktree changes.
- [ ] Run targeted ripgrep checks for obsolete paths and root-document links.
- [ ] Run `python -m pytest --collect-only -q` to confirm pytest accepts the revised exclusion path.
