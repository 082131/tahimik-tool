import json
import pytest
from src.data.noise_policy import (
    ProbabilityManifest,
    CategoryProbability,
    SyntheticPairLineage,
    build_probability_manifest,
)
from src.data.noise_generator import TagalogNoiseGenerator, ABBREVIATIONS, SLANG_MAP


def test_manifest_resolves_only_from_training_labels():
    # Training set has 10 samples, 4 with abbreviations
    training_labels = [{"categories": ["abbreviation"]} for _ in range(4)] + [
        {"categories": []} for _ in range(6)
    ]
    # Validation / test labels with different rates should NOT affect training manifest
    bounds = {
        "abbreviation": (0.1, 0.5),
        "orthographic": (0.05, 0.3),
    }

    manifest = build_probability_manifest(
        training_labels=training_labels,
        bounds=bounds,
        seed=42,
        source_split="train",
        resource_versions={"reviewed_lexicon": "2026.09"},
    )

    # Observed rate: 4/10 = 0.40
    cat_abbrev = manifest.categories["abbreviation"]
    assert cat_abbrev.positive_count == 4
    assert cat_abbrev.total_count == 10
    assert cat_abbrev.observed_rate == pytest.approx(0.40)
    assert cat_abbrev.resolved_probability == pytest.approx(0.40)

    # Orthographic was 0 in training, lower bound is 0.05
    cat_ortho = manifest.categories["orthographic"]
    assert cat_ortho.observed_rate == pytest.approx(0.0)
    assert cat_ortho.resolved_probability == pytest.approx(0.05)


def test_manifest_clamping_to_bounds():
    # Observed 0.80, but upper bound is 0.45
    training_labels = [{"categories": ["elongation"]} for _ in range(8)] + [
        {"categories": []} for _ in range(2)
    ]
    bounds = {"elongation": (0.10, 0.45)}

    manifest = build_probability_manifest(
        training_labels=training_labels,
        bounds=bounds,
        seed=42,
        resource_versions={"reviewed_lexicon": "2026.09"},
    )
    assert manifest.categories["elongation"].observed_rate == pytest.approx(0.80)
    assert manifest.categories["elongation"].resolved_probability == pytest.approx(0.45)


def test_manifest_canonical_hashing_and_readiness():
    training_labels = [{"categories": ["abbreviation"]} for _ in range(3)]
    bounds = {"abbreviation": (0.0, 1.0)}

    resources = {"reviewed_lexicon": "2026.09"}
    m1 = build_probability_manifest(training_labels, bounds, seed=42, resource_versions=resources)
    m2 = build_probability_manifest(training_labels, bounds, seed=42, resource_versions=resources)

    assert m1.manifest_id == m2.manifest_id
    assert len(m1.manifest_id) == 64  # SHA-256 hex string

    # Missing category bounds make readiness fail
    unready_manifest = ProbabilityManifest(
        schema_version="1.0.0",
        manifest_id="dummy",
        source_split="train",
        split_fingerprint="abc",
        seed=42,
        categories={},
        resource_versions={},
        created_at="2026-09-08T00:00:00Z",
        readiness_state="draft",
    )
    with pytest.raises(ValueError, match="not ready"):
        unready_manifest.require_ready()


def test_manifest_rejects_tampered_or_incomplete_reporting_contract():
    """Catches a fail-open manifest that could make an ineligible Stage 1 run look valid."""
    manifest = build_probability_manifest(
        [{"categories": ["abbreviation"]}],
        {"abbreviation": (0.0, 1.0)},
        resource_versions={"reviewed_lexicon": "2026.09"},
    )
    tampered = manifest.to_dict()
    tampered["categories"]["abbreviation"]["upper_bound"] = 2.0
    with pytest.raises(ValueError):
        ProbabilityManifest.from_dict(tampered).require_ready()

    incomplete = manifest.to_dict()
    incomplete.pop("readiness_state")
    with pytest.raises(ValueError):
        ProbabilityManifest.from_dict(incomplete).require_ready()


def test_no_identity_alternatives_in_dictionaries():
    for word, replacements in ABBREVIATIONS.items():
        assert word not in replacements, f"Identity alternative found in abbreviations: {word} -> {word}"

    for word, replacement in SLANG_MAP.items():
        assert word != replacement, f"Identity alternative found in SLANG_MAP: {word} -> {word}"


def test_synthetic_generation_never_produces_identical_pair():
    manifest = build_probability_manifest(
        [{"categories": ["abbreviation"]}],
        {"abbreviation": (0.0, 1.0)},
        resource_versions={"reviewed_lexicon": "2026.09"},
    )
    generator = TagalogNoiseGenerator(seed=123, manifest=manifest)
    text = "Magandang umaga sa lahat ng tao dito sa Pilipinas!"

    for _ in range(50):
        noisy, clean, lineage = generator.generate_pair(text)
        assert noisy != clean, f"Synthetic generation produced identical pair: '{noisy}' == '{clean}'"
        assert isinstance(lineage, SyntheticPairLineage)
        assert len(lineage.applied_correctable_categories) > 0


def test_generate_pair_rejects_text_that_cannot_be_corrupted():
    """Catches the old fallback that labelled an unchanged one-character input as char_swap."""
    manifest = build_probability_manifest(
        [{"categories": ["char_swap"]}],
        {"char_swap": (0.0, 1.0)},
        resource_versions={"reviewed_lexicon": "2026.09"},
    )
    generator = TagalogNoiseGenerator(seed=1, manifest=manifest)
    with pytest.raises(ValueError, match="distinct synthetic pair"):
        generator.generate_pair("a")
