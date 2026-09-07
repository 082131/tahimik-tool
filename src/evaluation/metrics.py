import re
from collections import Counter
from typing import Dict, List, Optional

import editdistance
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

        scores = []
        for order in range(1, 5):
            pred_ngrams = self._ngrams(pred_tokens, order)
            ref_ngrams = self._ngrams(ref_tokens, order)

            if not pred_ngrams:
                continue

            # Calculate precision & recall between prediction and ground-truth reference
            overlap = sum((pred_ngrams & ref_ngrams).values())
            total_pred = sum(pred_ngrams.values())
            total_ref = sum(ref_ngrams.values())

            precision = overlap / max(total_pred, 1)
            recall = overlap / max(total_ref, 1)

            # Apply penalty only for source n-grams absent from the reference.
            # N-grams that appear in both source and reference are correct to keep,
            # so they should not be penalised (matches shotakoyama/gleu sdiff logic).
            if source_tokens:
                source_ngrams = self._ngrams(source_tokens, order)
                # Keep only source n-grams the reference does NOT contain
                source_only = Counter({
                    ng: cnt
                    for ng, cnt in source_ngrams.items()
                    if ref_ngrams.get(ng, 0) == 0
                })
                bad_copies = sum((pred_ngrams & source_only).values())
                penalty = max(1.0 - (bad_copies / max(total_pred, 1)), 0.0)
            else:
                penalty = 1.0

            # Combined order score
            scores.append(min(precision, recall) * penalty)

        if not scores:
            return 0.0

        return 100.0 * (sum(scores) / len(scores))

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
    def _err(pred: str, ref: str, noisy: str) -> float:
        """Compute Error Reduction Rate (ERR) for a single sentence.

        Measures how much of the original edit distance to reference was eliminated:
            ERR = (dist(noisy, ref) - dist(pred, ref)) / dist(noisy, ref)

        Returns:
            1.0 if both noisy and pred already match ref,
            0.0 if noisy matches ref but pred introduced errors,
            or the normalized ratio of reduced edit distance.
        """
        before = editdistance.eval(noisy, ref)
        after = editdistance.eval(pred, ref)

        if before == 0:
            return 1.0 if after == 0 else 0.0

        return (before - after) / before

    def compute_err(
        self,
        predictions: List[str],
        references: List[str],
        noisy_inputs: List[str],
    ) -> float:
        """Compute average Error Reduction Rate across the dataset."""
        values = [
            self._err(p, r, n)
            for p, r, n in zip(predictions, references, noisy_inputs)
        ]
        return sum(values) / max(len(values), 1)

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
    def _alpha(pred: str, ref: str) -> float:
        """Compute word accuracy strictly for alphabetic words with sequence alignment."""
        pred_words = NormalizationMetrics._alphabetic_words(pred)
        ref_words = NormalizationMetrics._alphabetic_words(ref)

        if not ref_words:
            return 1.0 if not pred_words else 0.0

        dist = NormalizationMetrics._word_levenshtein(pred_words, ref_words)
        return max(0.0, 1.0 - dist / len(ref_words))


    def compute_alpha_word_accuracy(
        self,
        predictions: List[str],
        references: List[str],
    ) -> float:
        """Compute mean alphabetic word accuracy across all predictions."""
        values = [
            self._alpha(p, r)
            for p, r in zip(predictions, references)
        ]
        return sum(values) / max(len(values), 1)

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
        per_sentence = self.compute_per_sentence(predictions, references, noisy_inputs)
        return {
            metric: float(sum(scores) / max(len(scores), 1))
            for metric, scores in per_sentence.items()
        }
