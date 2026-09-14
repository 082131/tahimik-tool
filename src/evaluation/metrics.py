import math
import re
from collections import Counter
from typing import Dict, List, Optional

from sacrebleu.metrics import CHRF


class NormalizationMetrics:
    """Evaluation suite for text normalization tasks.

    Computes standard normalization metrics:
      - GLEU+: Generalized Language Evaluation Understanding with source penalty
      - chrF: Character n-gram F-score (via SacreBLEU)
      - ERR: Error Reduction Rate based on Levenshtein edit distance
      - Alpha Word Accuracy: Accuracy evaluated strictly on alphabetic tokens
    """

    def __init__(self):
        # Plain chrF: character 6-grams only, no word n-grams (word_order=0)
        self.chrf_scorer = CHRF(char_order=6, word_order=0)

    @staticmethod
    def _ngrams(tokens: List[str], order: int) -> Counter:
        """Extract n-grams of a given order from a list of tokens."""
        return Counter(
            tuple(tokens[i : i + order])
            for i in range(len(tokens) - order + 1)
        )

    def _sentence_gleu_plus(
        self,
        prediction: str,
        reference: str,
        noisy: str = "",
    ) -> float:
        """Calculate sentence-level GLEU+ score (0 to 100).

        GLEU evaluates n-gram overlaps between prediction and reference (orders 1-4),
        penalizing predictions that keep unchanged error patterns from the noisy source.
        """
        pred_tokens = prediction.split()
        ref_tokens = reference.split()
        source_tokens = noisy.split()

        # Handle empty string corner cases
        if not pred_tokens or not ref_tokens:
            return 100.0 if pred_tokens == ref_tokens else 0.0

        modified_precisions = []
        for order in range(1, 5):
            pred_ngrams = self._ngrams(pred_tokens, order)
            ref_ngrams = self._ngrams(ref_tokens, order)

            if not pred_ngrams:
                continue

            # Manuscript modified precision: correct reference overlap minus
            # source material retained beyond what the reference supports.
            overlap = sum((pred_ngrams & ref_ngrams).values())
            total_pred = sum(pred_ngrams.values())
            bad_copies = 0
            if source_tokens:
                source_ngrams = self._ngrams(source_tokens, order)
                for ngram, predicted_count in pred_ngrams.items():
                    copied_count = min(predicted_count, source_ngrams[ngram])
                    reference_count = min(predicted_count, ref_ngrams[ngram])
                    bad_copies += max(0, copied_count - reference_count)

            modified_precision = (overlap - bad_copies) / total_pred
            if modified_precision <= 0:
                return 0.0
            modified_precisions.append(modified_precision)

        if not modified_precisions:
            return 0.0

        geometric_mean = math.exp(
            sum(math.log(score) for score in modified_precisions)
            / len(modified_precisions)
        )
        candidate_length = len(pred_tokens)
        reference_length = len(ref_tokens)
        brevity_penalty = (
            1.0
            if candidate_length > reference_length
            else math.exp(1.0 - reference_length / candidate_length)
        )
        return 100.0 * brevity_penalty * geometric_mean

    def compute_gleu_plus(
        self,
        predictions: List[str],
        references: List[str],
        noisy_inputs: Optional[List[str]] = None,
    ) -> List[float]:
        """Compute sentence-level GLEU+ scores for a batch of predictions."""
        if noisy_inputs is None:
            noisy_inputs = [""] * len(predictions)

        return [
            self._sentence_gleu_plus(pred, ref, noisy)
            for pred, ref, noisy in zip(predictions, references, noisy_inputs)
        ]

    def compute_chrf(
        self,
        predictions: List[str],
        references: List[str],
    ) -> float:
        """Compute corpus-level chrF score using SacreBLEU."""
        return self.chrf_scorer.corpus_score(predictions, [references]).score

    @staticmethod
    def _evaluation_tokens(text: str) -> List[str]:
        """Return case-insensitive whitespace tokens with punctuation retained."""
        return text.casefold().split()

    @classmethod
    def _correct_token_count(cls, prediction: str, reference: str) -> tuple[int, int]:
        """Return aligned correct-token and reference-token counts for ERR."""
        predicted_tokens = cls._evaluation_tokens(prediction)
        reference_tokens = cls._evaluation_tokens(reference)
        distance = cls._word_levenshtein(predicted_tokens, reference_tokens)
        return max(0, len(reference_tokens) - distance), len(reference_tokens)

    @classmethod
    def _err(cls, pred: str, ref: str, noisy: str) -> float:
        """Compute sentence ERR from token accuracy over Leave-As-Is."""
        system_correct, reference_total = cls._correct_token_count(pred, ref)
        baseline_correct, _ = cls._correct_token_count(noisy, ref)

        if reference_total == 0:
            return 1.0 if not cls._evaluation_tokens(pred) else 0.0

        system_accuracy = system_correct / reference_total
        baseline_accuracy = baseline_correct / reference_total
        available_error = 1.0 - baseline_accuracy
        if available_error == 0:
            return 1.0 if system_accuracy == 1.0 else 0.0
        return (system_accuracy - baseline_accuracy) / available_error

    def compute_err(
        self,
        predictions: List[str],
        references: List[str],
        noisy_inputs: List[str],
    ) -> float:
        """Compute corpus ERR from pooled token totals before division."""
        system_correct = baseline_correct = reference_total = 0
        for prediction, reference, noisy in zip(predictions, references, noisy_inputs):
            correct_system, total = self._correct_token_count(prediction, reference)
            correct_baseline, _ = self._correct_token_count(noisy, reference)
            system_correct += correct_system
            baseline_correct += correct_baseline
            reference_total += total

        if reference_total == 0:
            return 1.0 if not predictions else 0.0

        system_accuracy = system_correct / reference_total
        baseline_accuracy = baseline_correct / reference_total
        available_error = 1.0 - baseline_accuracy
        if available_error == 0:
            return 1.0 if system_accuracy == 1.0 else 0.0
        return (system_accuracy - baseline_accuracy) / available_error

    @staticmethod
    def _alphabetic_words(text: str) -> List[str]:
        """Extract purely alphabetic words, stripped of surrounding punctuation."""
        words = text.split()
        pattern = re.compile(r"^[^\W\d_]+$", re.UNICODE)
        result = []
        for word in words:
            cleaned = word.strip(".,!?;:\"'()[]{}«»-–—").lower()
            if cleaned and pattern.match(cleaned):
                result.append(cleaned)
        return result

    @staticmethod
    def _word_levenshtein(seq1: List[str], seq2: List[str]) -> int:
        """Computes word sequence Levenshtein distance (insertions, deletions, substitutions)."""
        m, n = len(seq1), len(seq2)
        dp = [list(range(n + 1))] + [[i] + [0] * n for i in range(1, m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return dp[m][n]

    @staticmethod
    def _alpha_counts(pred: str, ref: str) -> tuple[int, int]:
        """Return aligned correct and reference alpha-word counts."""
        pred_words = NormalizationMetrics._alphabetic_words(pred)
        ref_words = NormalizationMetrics._alphabetic_words(ref)

        if not ref_words:
            return 0, 0

        dist = NormalizationMetrics._word_levenshtein(pred_words, ref_words)
        return max(0, len(ref_words) - dist), len(ref_words)

    @staticmethod
    def _alpha(pred: str, ref: str) -> float:
        """Compute alpha-word accuracy as a sentence-level percentage."""
        correct, total = NormalizationMetrics._alpha_counts(pred, ref)
        if total == 0:
            return 100.0 if not NormalizationMetrics._alphabetic_words(pred) else 0.0
        return 100.0 * correct / total


    def compute_alpha_word_accuracy(
        self,
        predictions: List[str],
        references: List[str],
    ) -> float:
        """Compute corpus alpha-word accuracy as a percentage."""
        correct = total = 0
        for prediction, reference in zip(predictions, references):
            sentence_correct, sentence_total = self._alpha_counts(prediction, reference)
            correct += sentence_correct
            total += sentence_total
        if total == 0:
            return 100.0 if not predictions else 0.0
        return 100.0 * correct / total

    def compute_per_sentence(
        self,
        predictions: List[str],
        references: List[str],
        noisy_inputs: List[str],
    ) -> Dict[str, List[float]]:
        """Compute all metrics on a per-sentence breakdown basis."""
        return {
            "gleu_plus": self.compute_gleu_plus(predictions, references, noisy_inputs),
            "chrf": [
                self.chrf_scorer.sentence_score(p, [r]).score
                for p, r in zip(predictions, references)
            ],
            "err": [
                self._err(p, r, n)
                for p, r, n in zip(predictions, references, noisy_inputs)
            ],
            "alpha_word_accuracy": [
                self._alpha(p, r)
                for p, r in zip(predictions, references)
            ],
        }

    def compute_all(
        self,
        predictions: List[str],
        references: List[str],
        noisy_inputs: List[str],
    ) -> Dict[str, float]:
        """Compute dataset-level summary averages for all normalization metrics."""
        return {
            "gleu_plus": float(sum(self.compute_gleu_plus(predictions, references, noisy_inputs)) / max(len(predictions), 1)),
            "chrf": self.compute_chrf(predictions, references),
            "err": self.compute_err(predictions, references, noisy_inputs),
            "alpha_word_accuracy": self.compute_alpha_word_accuracy(predictions, references),
        }
