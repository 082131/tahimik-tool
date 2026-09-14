# Data preprocessing and pipeline utilities: loading, cleaning, splitting, and pair generation.


import json
import csv
import re
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, List, Tuple, Dict, Optional, Sequence
from transformers import AutoTokenizer

from src.data.noise_label import compute_noise_level
from src.data.noise_generator import TagalogNoiseGenerator
from src.data.noise_policy import ProbabilityManifest, SyntheticPairLineage
from src.utils.logging_utils import setup_logger

logger = setup_logger("tahimik.data")


@dataclass(frozen=True)
class PairPreparationAudit:
    """Aggregate, non-content audit trail for paired length preparation."""

    total_pairs: int = 0
    unchanged_pairs: int = 0
    transformed_pairs: int = 0
    excluded_pairs: int = 0
    exclusions: List[Tuple[int, str]] = field(default_factory=list)


@dataclass(frozen=True)
class PreparedPairs:
    """Length-safe pairs and labels, ready to be split without leakage."""

    noisy_texts: List[str]
    clean_texts: List[str]
    noise_levels: List[float]
    source_indices: List[int]
    audit: PairPreparationAudit


def _token_length(tokenizer: AutoTokenizer, text: str) -> int:
    """Return the active tokenizer length without allowing truncation."""
    encoding = tokenizer(text, padding=False, truncation=False)
    token_ids = encoding["input_ids"]
    if hasattr(token_ids, "shape"):
        return int(token_ids.shape[-1])
    if token_ids and isinstance(token_ids[0], (list, tuple)):
        return len(token_ids[0])
    return len(token_ids)


def _word_spans(text: str) -> List[Tuple[str, int, int]]:
    return [(match.group(), match.start(), match.end()) for match in re.finditer(r"\S+", text)]


def _word_alignment(
    noisy_words: Sequence[str], clean_words: Sequence[str]
) -> List[Tuple[str, List[int], List[int]]]:
    """Align word positions with deterministic Levenshtein backtracking.

    Equal words and substitutions are anchors.  Adjacent insertions/deletions
    are attached to a neighbouring substitution so word splits and merges stay
    atomic, while consecutive substitutions remain distinct editable groups.
    """
    n, m = len(noisy_words), len(clean_words)
    costs = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        costs[i][0] = i
    for j in range(1, m + 1):
        costs[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            substitution = costs[i - 1][j - 1] + (noisy_words[i - 1] != clean_words[j - 1])
            costs[i][j] = min(substitution, costs[i - 1][j] + 1, costs[i][j - 1] + 1)

    operations: List[Tuple[str, List[int], List[int]]] = []
    i, j = n, m
    while i or j:
        if i and j:
            substitution_cost = costs[i - 1][j - 1] + (noisy_words[i - 1] != clean_words[j - 1])
            if costs[i][j] == substitution_cost:
                operations.append(("equal" if noisy_words[i - 1] == clean_words[j - 1] else "substitute", [i - 1], [j - 1]))
                i -= 1
                j -= 1
                continue
        if i and costs[i][j] == costs[i - 1][j] + 1:
            operations.append(("delete", [i - 1], []))
            i -= 1
        else:
            operations.append(("insert", [], [j - 1]))
            j -= 1
    operations.reverse()

    groups: List[Tuple[str, List[int], List[int]]] = []
    pending: List[Tuple[str, List[int], List[int]]] = []
    for operation in operations:
        if operation[0] in {"equal", "substitute"}:
            kind, noisy_indices, clean_indices = operation
            if pending and kind == "substitute":
                for _, pending_noisy, pending_clean in pending:
                    noisy_indices = pending_noisy + noisy_indices
                    clean_indices = pending_clean + clean_indices
                pending = []
            groups.append((kind, noisy_indices, clean_indices))
        elif groups and groups[-1][0] == "substitute":
            kind, noisy_indices, clean_indices = groups[-1]
            groups[-1] = (kind, noisy_indices + operation[1], clean_indices + operation[2])
        else:
            pending.append(operation)
    if pending:
        if groups and groups[-1][0] == "substitute":
            kind, noisy_indices, clean_indices = groups[-1]
            for _, pending_noisy, pending_clean in pending:
                noisy_indices += pending_noisy
                clean_indices += pending_clean
            groups[-1] = (kind, noisy_indices, clean_indices)
        else:
            groups.extend(pending)
    return groups


def prepare_paired_examples(
    noisy_texts: List[str],
    clean_texts: List[str],
    tokenizer: AutoTokenizer,
    max_input_length: int,
    max_target_length: int,
) -> PreparedPairs:
    """Make paired examples length-safe before labels and split assignment.

    The longest ordered prefix of complete aligned word groups that fits both
    tokenizer limits is retained.  A group that would overflow either side is
    excluded from both texts; no text is tokenized with truncation.
    """
    if len(noisy_texts) != len(clean_texts):
        raise ValueError("noisy_texts and clean_texts must have the same length")

    prepared_noisy: List[str] = []
    prepared_clean: List[str] = []
    noise_levels: List[float] = []
    source_indices: List[int] = []
    exclusions: List[Tuple[int, str]] = []
    unchanged = transformed = 0

    for source_index, (noisy, clean) in enumerate(zip(noisy_texts, clean_texts)):
        noisy_spans, clean_spans = _word_spans(noisy), _word_spans(clean)
        groups = _word_alignment(
            [word for word, _, _ in noisy_spans], [word for word, _, _ in clean_spans]
        )
        noisy_end = clean_end = 0
        kept_any = False
        for _, noisy_indices, clean_indices in groups:
            candidate_noisy_end = max((noisy_spans[index][2] for index in noisy_indices), default=noisy_end)
            candidate_clean_end = max((clean_spans[index][2] for index in clean_indices), default=clean_end)
            candidate_noisy = noisy[:candidate_noisy_end].rstrip()
            candidate_clean = clean[:candidate_clean_end].rstrip()
            if (
                _token_length(tokenizer, candidate_noisy) > max_input_length
                or _token_length(tokenizer, candidate_clean) > max_target_length
            ):
                break
            noisy_end, clean_end = candidate_noisy_end, candidate_clean_end
            kept_any = True

        if not kept_any:
            exclusions.append((source_index, "no_complete_aligned_word_group_fits_limit"))
            continue

        retained_noisy, retained_clean = noisy[:noisy_end].rstrip(), clean[:clean_end].rstrip()
        prepared_noisy.append(retained_noisy)
        prepared_clean.append(retained_clean)
        noise_levels.append(compute_noise_level(retained_noisy, retained_clean))
        source_indices.append(source_index)
        if retained_noisy == noisy and retained_clean == clean:
            unchanged += 1
        else:
            transformed += 1

    audit = PairPreparationAudit(
        total_pairs=len(noisy_texts),
        unchanged_pairs=unchanged,
        transformed_pairs=transformed,
        excluded_pairs=len(exclusions),
        exclusions=exclusions,
    )
    logger.info(
        "Prepared paired examples: %d unchanged, %d transformed, %d excluded",
        audit.unchanged_pairs,
        audit.transformed_pairs,
        audit.excluded_pairs,
    )
    return PreparedPairs(prepared_noisy, prepared_clean, noise_levels, source_indices, audit)


class DataPipeline:
    """
    Manages the full data preparation pipeline for TAHIMIK.

    Responsibilities:
        1. Load raw text data (CSV/JSON/TXT)
        2. Clean and validate sentences
        3. Generate synthetic noisy-clean pairs (Stage 1)
        4. Compute noise levels (n*) for all pairs
        5. Split into train/val/test with stratification
        6. Prepare for DataLoader consumption

    Args:
        config: A BaseConfig (or subclass) with data split ratios and paths.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        config=None,
        seed: int = 42,
        noise_manifest: Optional[ProbabilityManifest] = None,
    ):
        self.config = config
        self.rng = random.Random(seed)
        self.noise_manifest = noise_manifest
        self.noise_gen = (
            TagalogNoiseGenerator(seed=seed, manifest=noise_manifest)
            if noise_manifest is not None else None
        )
        self.synthetic_lineage: List[SyntheticPairLineage] = []
        self.synthetic_diagnostics: Dict[str, Any] = {}

    # --- Data Loading --------------------------------------------------------

    def load_clean_corpus(self, filepath: str) -> List[str]:
        """
        Load a clean Tagalog/Taglish corpus for synthetic noise injection.

        Supports .txt (one sentence per line), .csv (column 'text'),
        and .json (list of strings or list of {"text": ...}).

        Args:
            filepath: Path to the clean corpus file.

        Returns:
            List of clean sentences.
        """
        path = Path(filepath)
        sentences = []

        if path.suffix == ".txt":
            with open(path, "r", encoding="utf-8") as f:
                sentences = [line.strip() for line in f if line.strip()]

        elif path.suffix == ".csv":
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text = row.get("text", row.get("sentence", "")).strip()
                    if text:
                        sentences.append(text)

        elif path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        sentences.append(item.strip())
                    elif isinstance(item, dict):
                        text = item.get("text", item.get("sentence", ""))
                        if text:
                            sentences.append(text.strip())

        # Filter short sentences. Paired length preparation handles maximum
        # tokenizer lengths consistently after synthetic noise is generated.
        sentences = [s for s in sentences if len(s.split()) >= 4]

        logger.info(f"Loaded {len(sentences)} clean sentences from {filepath}")
        return sentences

    def load_gold_standard(
        self, filepath: str
    ) -> Tuple[List[str], List[str]]:
        """
        Load annotated noisy-clean sentence pairs (gold standard dataset).

        Expects a CSV/JSON file with 'noisy' and 'clean' columns/fields.

        Args:
            filepath: Path to the gold standard dataset.

        Returns:
            Tuple of (noisy_texts, clean_texts).
        """
        path = Path(filepath)
        noisy_texts = []
        clean_texts = []

        if path.suffix == ".csv":
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    noisy = row.get("noisy", row.get("input", "")).strip()
                    clean = row.get("clean", row.get("target", row.get("normalized", ""))).strip()
                    if noisy and clean:
                        noisy_texts.append(noisy)
                        clean_texts.append(clean)

        elif path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                noisy = item.get("noisy", item.get("input", "")).strip()
                clean = item.get("clean", item.get("target", item.get("normalized", ""))).strip()
                if noisy and clean:
                    noisy_texts.append(noisy)
                    clean_texts.append(clean)

        normalized = {}
        for noisy, clean in zip(noisy_texts, clean_texts):
            noisy, clean = self.clean_text(noisy), self.clean_text(clean)
            if noisy in normalized and normalized[noisy] != clean:
                raise ValueError(f"conflicting clean targets for noisy sentence: {noisy!r}")
            normalized[noisy] = clean
        noisy_texts = list(normalized.keys())
        clean_texts = list(normalized.values())
        logger.info(f"Loaded {len(noisy_texts)} gold standard pairs from {filepath}")
        return noisy_texts, clean_texts

    # --- Cleaning ------------------------------------------------------------

    def clean_text(self, text: str) -> str:
        """
        Normalize text: replace user mentions with @ANON, URLs with <URL>,
        and collapse duplicate whitespace.
        """
        # Anonymize mentions
        text = _replace_mentions(text)
        # Replace URLs
        text = _replace_urls(text)
        # Normalize whitespace
        text = " ".join(text.split())
        return text.strip()

    # --- Synthetic Data Generation -------------------------------------------

    def generate_synthetic_pairs(
        self,
        clean_sentences: List[str],
        tokenizer: AutoTokenizer,
        max_input_length: int,
        max_target_length: int,
        target_size: int = 1_000_000,
    ) -> Tuple[List[str], List[str], List[float]]:
        """
        Generate synthetic noisy-clean pairs for Stage 1 pretraining.

        Applies the noise generator to clean sentences, cycling through
        the corpus multiple times if needed to reach the target size.

        Args:
            clean_sentences: Clean Tagalog/Taglish sentences.
            target_size: Desired number of synthetic pairs.

        Returns:
            Tuple of prepared (noisy_texts, clean_texts, noise_levels).
        """
        if self.noise_gen is None or self.noise_manifest is None:
            raise ValueError("Stage 1 synthetic generation requires a ready noise manifest")
        if not clean_sentences:
            raise ValueError("Cannot generate synthetic pairs from an empty clean corpus")

        noisy_texts = []
        clean_texts = []
        noise_levels = []
        self.synthetic_lineage = []

        logger.info(
            f"Generating {target_size} manifest-governed synthetic pairs"
        )

        for index in range(target_size):
            base = clean_sentences[index % len(clean_sentences)]
            try:
                noisy, clean, lineage = self.noise_gen.generate_pair(
                    base, pair_id=f"syn_{index:08d}", base_id=str(index % len(clean_sentences))
                )
            except ValueError as exc:
                raise ValueError(
                    f"Unable to generate requested synthetic pair {index + 1}/{target_size}: {exc}"
                ) from exc
            if noisy == clean:
                raise ValueError("Synthetic generation produced a copy pair")
            noisy_texts.append(noisy)
            clean_texts.append(clean)
            noise_levels.append(compute_noise_level(noisy, clean))
            self.synthetic_lineage.append(lineage)

        ordered_noise = sorted(noise_levels)
        def quantile(q: float) -> float:
            return ordered_noise[int((len(ordered_noise) - 1) * q)]
        category_counts: Dict[str, int] = {}
        for lineage in self.synthetic_lineage:
            for category in lineage.applied_preserved_categories + lineage.applied_correctable_categories:
                category_counts[category] = category_counts.get(category, 0) + 1
        self.synthetic_diagnostics = {
            "manifest_id": self.noise_manifest.manifest_id,
            "requested_count": target_size,
            "generated_count": len(noisy_texts),
            "mean_noise": sum(noise_levels) / len(noise_levels),
            "median_noise": quantile(0.5),
            "noise_quantiles": {"q25": quantile(0.25), "q75": quantile(0.75)},
            "zero_noise_fraction": sum(n == 0.0 for n in noise_levels) / len(noise_levels),
            "category_counts": category_counts,
        }

        prepared = prepare_paired_examples(
            noisy_texts,
            clean_texts,
            tokenizer=tokenizer,
            max_input_length=max_input_length,
            max_target_length=max_target_length,
        )
        logger.info(
            "Generated %d synthetic pairs; %d remain after paired length preparation",
            len(noisy_texts),
            len(prepared.noisy_texts),
        )
        return prepared.noisy_texts, prepared.clean_texts, prepared.noise_levels

    # --- Splitting -----------------------------------------------------------

    def split_data(
        self,
        noisy_texts: List[str],
        clean_texts: List[str],
        noise_levels: List[float],
        train_ratio: float,
        val_ratio: float,
        test_ratio: float = 0.0,
    ) -> Dict[str, Tuple[List[str], List[str], List[float]]]:
        """
        Split data into train/val/test partitions with shuffling.

        Args:
            noisy_texts, clean_texts, noise_levels: Parallel lists.
            train_ratio, val_ratio, test_ratio: Split proportions.

        Returns:
            Dict with keys "train", "val", and optionally "test",
            each mapping to (noisy, clean, noise_levels) tuples.
        """
        n = len(noisy_texts)
        indices = list(range(n))
        self.rng.shuffle(indices)

        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        splits = {}

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]

        splits["train"] = (
            [noisy_texts[i] for i in train_idx],
            [clean_texts[i] for i in train_idx],
            [noise_levels[i] for i in train_idx],
        )
        splits["val"] = (
            [noisy_texts[i] for i in val_idx],
            [clean_texts[i] for i in val_idx],
            [noise_levels[i] for i in val_idx],
        )

        if test_ratio > 0:
            test_idx = indices[val_end:]
            splits["test"] = (
                [noisy_texts[i] for i in test_idx],
                [clean_texts[i] for i in test_idx],
                [noise_levels[i] for i in test_idx],
            )

        for split_name, (noisy, clean, nl) in splits.items():
            logger.info(f"  {split_name}: {len(noisy)} pairs")

        return splits


# --- Helper functions ----------------------------------------------------

def _replace_mentions(text: str) -> str:
    """Replace @username patterns with @ANON."""
    return re.sub(r"@\w+", "@ANON", text)


def _replace_urls(text: str) -> str:
    """Replace URLs with <URL>."""
    url_pattern = r"https?://\S+|www\.\S+"
    return re.sub(url_pattern, "<URL>", text)





