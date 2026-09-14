import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CategoryProbability:
    category: str
    group: str  # "preserved_augmentation" or "correctable_noise"
    positive_count: int
    total_count: int
    observed_rate: float
    lower_bound: float
    upper_bound: float
    resolved_probability: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SyntheticPairLineage:
    pair_id: str
    base_sentence_id: str
    manifest_id: str
    resource_versions: Dict[str, str]
    applied_preserved_categories: List[str]
    applied_correctable_categories: List[str]
    seed_derivation: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProbabilityManifest:
    schema_version: str
    manifest_id: str
    source_split: str
    split_fingerprint: str
    seed: int
    categories: Dict[str, CategoryProbability]
    resource_versions: Dict[str, str]
    created_at: str
    readiness_state: str = "draft"  # "ready" or "draft"

    def require_ready(self) -> None:
        """Fail-closed check ensuring all categories and resources are approved and ready."""
        if self.readiness_state != "ready":
            raise ValueError(f"ProbabilityManifest is not ready (state={self.readiness_state})")
        if self.source_split != "train":
            raise ValueError("ProbabilityManifest must be derived from the training split")
        if not self.manifest_id or not self.split_fingerprint:
            raise ValueError("ProbabilityManifest is missing its identity or training fingerprint")
        if not self.resource_versions:
            raise ValueError("ProbabilityManifest has no approved resource versions")
        if not self.categories:
            raise ValueError("ProbabilityManifest has no approved category records")
        for cat, prob in self.categories.items():
            expected_group = "preserved_augmentation" if cat in PRESERVED_CATEGORIES else "correctable_noise"
            if cat not in PRESERVED_CATEGORIES | CORRECTABLE_CATEGORIES or prob.category != cat:
                raise ValueError(f"Unknown or malformed category record: {cat}")
            if prob.group != expected_group:
                raise ValueError(f"Category {cat} has invalid group {prob.group!r}")
            if not (0.0 <= prob.lower_bound <= prob.upper_bound <= 1.0):
                raise ValueError(
                    f"Category {cat} has invalid probability bounds"
                )
            if not (0.0 <= prob.observed_rate <= 1.0 and 0.0 <= prob.resolved_probability <= 1.0):
                raise ValueError(f"Category {cat} has an invalid probability")
        if _canonical_digest(self.to_dict()) != self.manifest_id:
            raise ValueError("ProbabilityManifest ID does not match its canonical contents")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "source_split": self.source_split,
            "split_fingerprint": self.split_fingerprint,
            "seed": self.seed,
            "categories": {k: v.to_dict() for k, v in self.categories.items()},
            "resource_versions": self.resource_versions,
            "created_at": self.created_at,
            "readiness_state": self.readiness_state,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProbabilityManifest":
        categories = {
            k: CategoryProbability(**v)
            for k, v in data.get("categories", {}).items()
        }
        return cls(
            schema_version=data.get("schema_version", "1.0.0"),
            manifest_id=data.get("manifest_id", ""),
            source_split=data.get("source_split", "train"),
            split_fingerprint=data.get("split_fingerprint", ""),
            seed=data.get("seed", 42),
            categories=categories,
            resource_versions=data.get("resource_versions", {}),
            created_at=data.get("created_at", ""),
            readiness_state=data.get("readiness_state", "draft"),
        )


def load_probability_manifest(path: str) -> ProbabilityManifest:
    """Load and validate a saved reporting manifest before Stage 1 begins."""
    with open(path, "r", encoding="utf-8") as handle:
        manifest = ProbabilityManifest.from_dict(json.load(handle))
    manifest.require_ready()
    return manifest


PRESERVED_CATEGORIES = {"slang", "emoji", "code_switching", "taglish_morphology"}
CORRECTABLE_CATEGORIES = {
    "abbreviation",
    "orthographic",
    "elongation",
    "punctuation",
    "capitalization",
    "vowel_omission",
    "char_swap",
    "typo",
}


def _canonical_digest(data: Dict[str, Any]) -> str:
    """Computes deterministic SHA-256 over canonical JSON without manifest_id and created_at."""
    clean_data = {
        k: v for k, v in data.items() if k not in ("manifest_id", "created_at")
    }
    canonical_json = json.dumps(clean_data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()



def build_probability_manifest(
    training_labels: List[Dict[str, Any]],
    bounds: Dict[str, Tuple[float, float]],
    seed: int = 42,
    resource_versions: Optional[Dict[str, str]] = None,
    source_split: str = "train",
) -> ProbabilityManifest:
    """
    Derives category prevalence from training labels and approved bounds.
    """
    if source_split != "train":
        raise ValueError("Probability manifests may only be built from the training split")
    if not training_labels:
        raise ValueError("Training labels are required to build a reporting manifest")
    if not resource_versions:
        raise ValueError("Approved resource versions are required to build a reporting manifest")
    total_samples = len(training_labels)
    counts: Dict[str, int] = {}
    for item in training_labels:
        for cat in item.get("categories", []):
            counts[cat] = counts.get(cat, 0) + 1

    categories: Dict[str, CategoryProbability] = {}
    for cat, (lower, upper) in bounds.items():
        if cat not in PRESERVED_CATEGORIES | CORRECTABLE_CATEGORIES:
            raise ValueError(f"Unknown noise category: {cat}")
        if not (0.0 <= lower <= upper <= 1.0):
            raise ValueError(f"Category {cat} bounds must satisfy 0 <= lower <= upper <= 1")
        pos = counts.get(cat, 0)
        observed = (pos / total_samples) if total_samples > 0 else 0.0
        resolved = min(max(observed, lower), upper)

        group = (
            "preserved_augmentation"
            if cat in PRESERVED_CATEGORIES
            else "correctable_noise"
        )

        categories[cat] = CategoryProbability(
            category=cat,
            group=group,
            positive_count=pos,
            total_count=total_samples,
            observed_rate=observed,
            lower_bound=lower,
            upper_bound=upper,
            resolved_probability=resolved,
        )

    split_fingerprint = hashlib.sha256(
        json.dumps(
            {"source_split": source_split, "seed": seed, "training_labels": training_labels},
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]

    created_at = datetime.now(timezone.utc).isoformat()
    manifest_dict = {
        "schema_version": "1.0.0",
        "source_split": source_split,
        "split_fingerprint": split_fingerprint,
        "seed": seed,
        "categories": {k: v.to_dict() for k, v in categories.items()},
        "resource_versions": resource_versions,
        "created_at": created_at,
        "readiness_state": "ready",
    }
    manifest_id = _canonical_digest(manifest_dict)

    return ProbabilityManifest(
        schema_version="1.0.0",
        manifest_id=manifest_id,
        source_split=source_split,
        split_fingerprint=split_fingerprint,
        seed=seed,
        categories=categories,
        resource_versions=resource_versions,
        created_at=created_at,
        readiness_state="ready",
    )
