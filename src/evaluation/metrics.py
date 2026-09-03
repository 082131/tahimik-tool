import re
from collections import Counter
from typing import Dict, List, Optional
import editdistance
from sacrebleu.metrics import CHRF

class NormalizationMetrics:
    def __init__(self):
        self.chrf_scorer = CHRF(char_order=6, word_order=2)

    @staticmethod
    def _ngrams(tokens: List[str], order: int) -> Counter:
        return Counter(tuple(tokens[i:i + order]) for i in range(len(tokens) - order + 1))

    def _sentence_gleu_plus(self, prediction: str, reference: str, noisy: str = "") -> float:
        pred, ref, source = prediction.split(), reference.split(), noisy.split()
        if not pred or not ref:
            return 100.0 if pred == ref else 0.0
        scores = []
        for order in range(1, 5):
            pc, rc = self._ngrams(pred, order), self._ngrams(ref, order)
            if not pc:
                continue
            overlap = sum((pc & rc).values())
            precision = overlap / max(sum(pc.values()), 1)
            recall = overlap / max(sum(rc.values()), 1)
            source_overlap = sum((pc & self._ngrams(source, order)).values()) if source else 0
            penalty = 1.0 - source_overlap / max(sum(pc.values()), 1) if source else 1.0
            scores.append(min(precision, recall) * max(penalty, 0.0))
        return 100.0 * sum(scores) / max(len(scores), 1)

    def compute_gleu_plus(self, predictions: List[str], references: List[str], noisy_inputs: Optional[List[str]] = None) -> List[float]:
        noisy_inputs = noisy_inputs or [""] * len(predictions)
        return [self._sentence_gleu_plus(p, r, n) for p, r, n in zip(predictions, references, noisy_inputs)]

    def compute_chrf(self, predictions: List[str], references: List[str]) -> float:
        return self.chrf_scorer.corpus_score(predictions, [references]).score

    @staticmethod
    def _err(pred: str, ref: str, noisy: str) -> float:
        before, after = editdistance.eval(noisy, ref), editdistance.eval(pred, ref)
        return 1.0 if before == 0 and after == 0 else (0.0 if before == 0 else (before - after) / before)

    def compute_err(self, predictions: List[str], references: List[str], noisy_inputs: List[str]) -> float:
        values = [self._err(p, r, n) for p, r, n in zip(predictions, references, noisy_inputs)]
        return sum(values) / max(len(values), 1)

    @staticmethod
    def _alpha(pred: str, ref: str) -> float:
        pattern = re.compile(r"^[^\W\d_]+$", re.UNICODE)
        ref_words, pred_words = ref.split(), pred.split()
        indices = [i for i, word in enumerate(ref_words) if pattern.match(word)]
        return sum(i < len(pred_words) and pred_words[i].lower() == ref_words[i].lower() for i in indices) / max(len(indices), 1)

    def compute_alpha_word_accuracy(self, predictions: List[str], references: List[str]) -> float:
        values = [self._alpha(p, r) for p, r in zip(predictions, references)]
        return sum(values) / max(len(values), 1)

    def compute_per_sentence(self, predictions: List[str], references: List[str], noisy_inputs: List[str]) -> Dict[str, List[float]]:
        return {
            "gleu_plus": self.compute_gleu_plus(predictions, references, noisy_inputs),
            "chrf": [self.chrf_scorer.sentence_score(p, [r]).score for p, r in zip(predictions, references)],
            "err": [self._err(p, r, n) for p, r, n in zip(predictions, references, noisy_inputs)],
            "alpha_word_accuracy": [self._alpha(p, r) for p, r in zip(predictions, references)],
        }

    def compute_all(self, predictions: List[str], references: List[str], noisy_inputs: List[str]) -> Dict[str, float]:
        per_sentence = self.compute_per_sentence(predictions, references, noisy_inputs)
        return {key: float(sum(values) / max(len(values), 1)) for key, values in per_sentence.items()}
