import math

import pytest

from src.evaluation.metrics import NormalizationMetrics


def test_gleu_plus_returns_zero_when_a_modified_precision_is_zero():
    metrics = NormalizationMetrics()

    score = metrics.compute_gleu_plus(
        ["a b c d"],
        ["a b c x"],
        ["a b c d"],
    )[0]

    assert score == 0.0


def test_gleu_plus_applies_the_manuscript_brevity_penalty():
    metrics = NormalizationMetrics()

    score = metrics.compute_gleu_plus(
        ["a b c d"],
        ["a b c d e"],
        ["a b c d e"],
    )[0]

    assert score == pytest.approx(100 * math.exp(1 - 5 / 4))


def test_corpus_err_pools_token_totals_before_the_leave_as_is_ratio():
    metrics = NormalizationMetrics()

    score = metrics.compute_err(
        ["a b", "w x y z"],
        ["a b", "a b c d"],
        ["a x", "w x y z"],
    )

    assert score == pytest.approx(0.2)


def test_err_uses_case_insensitive_tokens_but_retains_punctuation():
    metrics = NormalizationMetrics()

    scores = metrics.compute_per_sentence(
        ["HELLO, yes!"],
        ["hello, yes!"],
        ["Hello, nope!"],
    )

    assert scores["err"] == [pytest.approx(1.0)]


def test_alpha_word_accuracy_reports_pooled_percentage_not_mean_sentence_ratio():
    metrics = NormalizationMetrics()

    score = metrics.compute_alpha_word_accuracy(
        ["a wrong", "one two three four five six seven eight"],
        ["a right", "one two three four five six seven eight"],
    )

    assert score == pytest.approx(90.0)


def test_compute_all_reports_corpus_chrf():
    metrics = NormalizationMetrics()
    predictions = ["a", "a b c d"]
    references = ["a b c d", "a"]
    noisy_inputs = references

    result = metrics.compute_all(predictions, references, noisy_inputs)

    assert result["chrf"] == pytest.approx(metrics.compute_chrf(predictions, references))
