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
    readiness_state: str = "ready"  # "ready" or "draft"

    def require_ready(self) -> None:
        """Fail-closed check ensuring all categories and resources are approved and ready."""
        if self.readiness_state != "ready":
            raise ValueError(f"ProbabilityManifest is not ready (state={self.readiness_state})")
        if not self.categories:
            raise ValueError("ProbabilityManifest has no approved category records")
        for cat, prob in self.categories.items():
            if prob.lower_bound > prob.upper_bound:
                raise ValueError(
                    f"Category {cat} invalid bounds: lower {prob.lower_bound} > upper {prob.upper_bound}"
                )

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
            readiness_state=data.get("readiness_state", "ready"),
        )


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
    total_samples = len(training_labels)
    counts: Dict[str, int] = {}
    for item in training_labels:
        for cat in item.get("categories", []):
            counts[cat] = counts.get(cat, 0) + 1

    categories: Dict[str, CategoryProbability] = {}
    for cat, (lower, upper) in bounds.items():
        if lower > upper:
            raise ValueError(f"Category {cat} lower bound {lower} exceeds upper bound {upper}")
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
        f"{source_split}:{total_samples}:{seed}".encode("utf-8")
    ).hexdigest()[:16]

    created_at = datetime.now(timezone.utc).isoformat()
    resource_versions = resource_versions or {"lexicon": "1.0.0"}

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
