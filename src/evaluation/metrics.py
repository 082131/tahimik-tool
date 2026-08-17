# =============================================================================
# Normalization Quality Metrics for TAHIMIK Evaluation
#
# Four metrics from the manuscript (Evaluation Criteria for Normalization
# Accuracy section):
#
#   GLEU+ — Sentence-level variant of BLEU using 1-4 n-grams with
#           add-one smoothing. Averaged across all test sentences.
#           Captures n-gram precision and recall at the byte/word level.
#
#   chrF  — Character-level F-score using character n-grams (1-6) and
#           word n-grams (0-2). Computed by SacreBLEU. Particularly
#           suitable for byte-level models since it evaluates at the
#           character grain, aligning with ByT5's input representation.
#
#   ERR   — Error Reduction Rate. Measures what fraction of the errors
#           present in the noisy input were corrected by the model:
#           ERR = (errors_before - errors_after) / errors_before
#           Uses edit distance. ERR > 0 means improvement over the input.
#
#   Alpha-word Accuracy — Fraction of alpha-only words in the predicted
#           output that exactly match the reference. Focuses on real word
#           accuracy, ignoring punctuation and special tokens.
# =============================================================================

import re
import editdistance
from typing import List, Dict
from sacrebleu.metrics import BLEU, CHRF


class NormalizationMetrics:
    """
    Computes all four normalization accuracy metrics.

    Usage:
        metrics = NormalizationMetrics()
        results = metrics.compute_all(predictions, references, noisy_inputs)
    """

    def __init__(self):
        # SacreBLEU BLEU scorer with sentence-level smoothing
        self.bleu_scorer = BLEU(smooth_method="add-k", smooth_value=1)
        # chrF scorer with character 6-grams and word 2-grams
        self.chrf_scorer = CHRF(char_order=6, word_order=2)

    def compute_gleu_plus(
        self,
        predictions: List[str],
        references: List[str],
    ) -> float:
        """
        Compute GLEU+ (sentence-level BLEU with add-one smoothing).

        Scores each prediction against its reference individually,
        then averages across all sentences.

        Returns:
            Average GLEU+ score in [0, 100].
        """
        scores = []
        for pred, ref in zip(predictions, references):
            score = self.bleu_scorer.sentence_score(pred, [ref])
            scores.append(score.score)

        return sum(scores) / max(len(scores), 1)

    def compute_chrf(
        self,
        predictions: List[str],
        references: List[str],
    ) -> float:
        """
        Compute corpus-level chrF score.

        Returns:
            chrF score in [0, 100].
        """
        result = self.chrf_scorer.corpus_score(predictions, [references])
        return result.score

    def compute_err(
        self,
        predictions: List[str],
        references: List[str],
        noisy_inputs: List[str],
    ) -> float:
        """
        Compute Error Reduction Rate (ERR).

        ERR = (errors_before - errors_after) / errors_before

        where errors_before = edit_distance(noisy, reference)
              errors_after  = edit_distance(prediction, reference)

        An ERR of 1.0 means all errors were corrected.
        An ERR of 0.0 means the model made no improvement.
        A negative ERR means the model introduced more errors.

        Returns:
            Average ERR across all sentences.
        """
        err_scores = []

        for pred, ref, noisy in zip(predictions, references, noisy_inputs):
            errors_before = editdistance.eval(noisy, ref)
            errors_after = editdistance.eval(pred, ref)

            if errors_before == 0:
                # Input was already clean — ERR is 1.0 if output is also
                # clean, otherwise penalize
                err = 1.0 if errors_after == 0 else 0.0
            else:
                err = (errors_before - errors_after) / errors_before

            err_scores.append(err)

        return sum(err_scores) / max(len(err_scores), 1)

    def compute_alpha_word_accuracy(
        self,
        predictions: List[str],
        references: List[str],
    ) -> float:
        """
        Compute alpha-word accuracy.

        For each sentence pair, extract alpha-only words (letters only,
        no digits/punctuation/emoji), align by position, and compute
        the fraction that match exactly (case-insensitive).

        Returns:
            Average alpha-word accuracy in [0, 1].
        """
        total_words = 0
        correct_words = 0

        alpha_pattern = re.compile(r"^[a-zA-Z]+$")

        for pred, ref in zip(predictions, references):
            ref_words = ref.split()
            pred_words = pred.split()

            for i, ref_word in enumerate(ref_words):
                if not alpha_pattern.match(ref_word):
                    continue

                total_words += 1
                if i < len(pred_words):
                    if pred_words[i].lower() == ref_word.lower():
                        correct_words += 1

        return correct_words / max(total_words, 1)

    def compute_all(
        self,
        predictions: List[str],
        references: List[str],
        noisy_inputs: List[str],
    ) -> Dict[str, float]:
        """
        Compute all four normalization metrics.

        Args:
            predictions: Model-generated normalized sentences.
            references: Gold-standard reference sentences.
            noisy_inputs: Original noisy input sentences.

        Returns:
            Dict with keys: 'gleu_plus', 'chrf', 'err', 'alpha_word_accuracy'.
        """
        return {
            "gleu_plus": self.compute_gleu_plus(predictions, references),
            "chrf": self.compute_chrf(predictions, references),
            "err": self.compute_err(predictions, references, noisy_inputs),
            "alpha_word_accuracy": self.compute_alpha_word_accuracy(
                predictions, references
            ),
        }
