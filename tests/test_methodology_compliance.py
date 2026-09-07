import json
from pathlib import Path

import numpy as np
import pytest

from src.data.noise_generator import TagalogNoiseGenerator
from src.data.preprocessing import DataPipeline
from src.evaluation.metrics import NormalizationMetrics
from src.evaluation.statistical_tests import StatisticalAnalysis
from src.utils.reproducibility import collect_run_metadata


def test_source_aware_gleu_plus_penalizes_copying_noisy_text():
    metrics = NormalizationMetrics()
    clean = "Ang sarap ng food!"
    noisy = "Ang sarapppp ng food!"
    corrected = "Ang sarap ng food!"
    copied = noisy
    good = metrics.compute_gleu_plus([corrected], [clean], [noisy])[0]
    bad = metrics.compute_gleu_plus([copied], [clean], [noisy])[0]
    assert good > bad


def test_bootstrap_is_two_tailed_add_one_and_ci_gated():
    analysis = StatisticalAnalysis(n_bootstrap=20, seed=1)
    result = analysis.paired_bootstrap([0.0, 0.0], [1.0, 1.0])
    assert result["p_value"] == pytest.approx(2 / 21)
    assert result["ci_lower"] > 0
    assert result["significant"] is False


def test_wilcoxon_reports_median_iqr_and_rank_biserial():
    result = StatisticalAnalysis().wilcoxon_test([1, 2, 3], [2, 3, 4])
    assert {"median_a", "iqr_a", "rank_biserial"} <= result.keys()


def test_holm_is_step_down_and_reports_adjusted_p():
    result = StatisticalAnalysis.holm_bonferroni([("a", 0.001), ("b", 0.04), ("c", 0.2)])
    assert result[0]["significant_corrected"] is True
    assert result[1]["significant_corrected"] is False
    assert "adjusted_p" in result[0]


def test_protected_synthetic_categories_are_not_applied():
    generator = TagalogNoiseGenerator(
        seed=7,
        probabilities={"slang": 1.0, "emoji": 1.0},
    )
    text = "Ang sarap ng food!"
    assert generator.apply_noise(text) == text


def test_gold_validation_rejects_conflicting_targets(tmp_path: Path):
    path = tmp_path / "gold.csv"
    path.write_text("noisy,clean\nkumusta,hello\nkumusta,hi\n", encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting"):
        DataPipeline().load_gold_standard(str(path))


def test_metadata_is_json_serializable():
    metadata = collect_run_metadata(seed=42, config={"x": 1})
    json.dumps(metadata)
    assert metadata["seed"] == 42
    assert "git_sha" in metadata and "dirty" in metadata


def test_alpha_word_accuracy_is_sequence_aligned_not_positional_zip():
    metrics = NormalizationMetrics()
    ref = "ang bilis tumakbo ng bata"  # 5 words
    # 1 insertion at beginning ("hoy"): words are shifted by 1
    pred = "hoy ang bilis tumakbo ng bata"
    score = metrics.compute_alpha_word_accuracy([pred], [ref])
    # With alignment: 1 edit out of 5 words = 4/5 = 0.80
    assert score >= 0.75

    # Case insensitivity and punctuation handling
    pred2 = "ANG BILIS tumakbo, ng bata!"
    assert metrics.compute_alpha_word_accuracy([pred2], [ref]) == pytest.approx(1.0)


