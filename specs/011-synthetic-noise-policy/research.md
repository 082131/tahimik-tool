# Research and Decisions

## Probability source

**Decision:** Count binary category labels only among IDs assigned to the gold training split. Resolve each probability with `min(max(observed, lower), upper)`.

**Rationale:** This avoids validation/test leakage while keeping the bounded policy directly auditable.

## Two-pass generation

**Decision:** Apply preserved augmentation to the clean base, copy that result to both sides, then apply correctable corruption only to the input.

**Rationale:** Slang, emoji, code-switching, and Taglish morphology remain legitimate target content while still being synthetically injected.

## Lexicon integration

**Decision:** Use a generic reviewed CSV provider now. A future KWF-specific provider requires a documented API and authorization.

**Rationale:** No verified public KWF API contract is available, and scraping creates permission, stability, and reproducibility risks.

## Readiness behavior

**Decision:** Missing bounds or required reviewed resources are explicit eligibility failures, not fallbacks to hardcoded probabilities.

**Alternatives rejected:** estimating from all gold labels, silently using defaults, or scraping a website.
