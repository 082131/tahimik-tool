import pytest

from src.evaluation.statistical_tests import StatisticalAnalysis


def test_wilcoxon_reports_signed_rank_effect_and_paired_difference_summaries_in_gb():
    result = StatisticalAnalysis().wilcoxon_test(
        [0.0, 0.0, 0.0],
        [1024.0, 2048.0, -3072.0],
    )

    assert result["median_difference_gb"] == pytest.approx(1.0)
    assert result["iqr_difference_gb"] == pytest.approx(2.5)
    assert result["rank_biserial"] == pytest.approx(0.0)


def test_efficiency_holm_correction_is_a_separate_two_test_family_per_baseline():
    per_sentence_scores = {
        name: {
            metric: [float(index), float(index + 1)]
            for metric in ("gleu_plus", "chrf", "err", "alpha_word_accuracy")
        }
        | {"inference_time": [1.0, 2.0]}
        for index, name in enumerate(("byt5", "mrt5", "tahimik"))
    }
    gpu_memory_runs = {
        "byt5": [3.0, 4.0],
        "mrt5": [3.0, 4.0],
        "tahimik": [1.0, 2.0],
    }

    result = StatisticalAnalysis(n_bootstrap=10, seed=42).run_full_comparison(
        per_sentence_scores,
        gpu_memory_runs,
    )

    families = result["holm_bonferroni"]["efficiency"]
    assert set(families) == {"byt5_vs_tahimik", "mrt5_vs_tahimik"}
    assert all(len(family) == 2 for family in families.values())
