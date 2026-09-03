# Research and Decisions

## Single builder

**Decision:** Every command and checkpoint uses the same `build_run_provenance` API with context-specific lineage fields.

**Rationale:** Shared required keys cannot drift between training, evaluation, benchmarking, and orchestration.

## Canonical serialization

**Decision:** Normalize dataclasses, paths, devices, dtypes, tuples, mappings, and sequences recursively; serialize JSON with sorted keys and fixed separators before SHA-256 hashing.

## CUDA reporting

**Decision:** Distinguish PyTorch CUDA build version, runtime availability, cuDNN version, device name/capability, and selected precision. Missing fields are explicit `null`, never guessed.

## Eligibility

**Decision:** Store `eligible` plus a sorted list of stable reason codes. Dirty worktrees and incomplete data contracts are ineligible; CPU specifically invalidates GPU-efficiency evidence.

**Alternatives rejected:** stringifying the whole configuration or maintaining separate metadata dictionaries per script.
