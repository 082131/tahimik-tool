# =============================================================================
# Data Preprocessing and Pipeline Management
#
# Handles loading, cleaning, splitting, and preparing data for the
# two-stage training pipeline:
#   Stage 1: Synthetic data pretraining (~1M pairs, 90/10 split)
#   Stage 2: Gold standard fine-tuning (~15K pairs, 80/10/10 split)
#
# The same pipeline is used for all three model variants to ensure
# the control variable (training data) is held constant.
# =============================================================================

import json
import csv
import re
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from transformers import AutoTokenizer

from src.data.noise_label import compute_noise_level
from src.data.noise_generator import TagalogNoiseGenerator
from src.utils.logging_utils import setup_logger

logger = setup_logger("tahimik.data")


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

    def __init__(self, config=None, seed: int = 42):
        self.config = config
        self.rng = random.Random(seed)
        self.noise_gen = TagalogNoiseGenerator(seed=seed)

    # â”€â”€ Data Loading â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

        # Filter by minimum word count (manuscript: â‰¥4 words)
        sentences = [s for s in sentences if len(s.split()) >= 4]

        # Filter by maximum byte length (manuscript: â‰¤1024 bytes)
        sentences = [s for s in sentences if len(s.encode("utf-8")) <= 1024]

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

    # â”€â”€ Cleaning â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def clean_text(self, text: str) -> str:
        """
        Apply basic text cleaning as described in the manuscript.

        Replaces usernames with @ANON and URLs with <URL>.
        Removes duplicate whitespace.
        """
        # Anonymize mentions
        text = _replace_mentions(text)
        # Replace URLs
        text = _replace_urls(text)
        # Normalize whitespace
        text = " ".join(text.split())
        return text.strip()

    # â”€â”€ Synthetic Data Generation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def generate_synthetic_pairs(
        self,
        clean_sentences: List[str],
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
            Tuple of (noisy_texts, clean_texts, noise_levels).
        """
        noisy_texts = []
        clean_texts = []
        noise_levels = []

        # How many noisy variants per clean sentence to reach target
        passes = max(1, target_size // len(clean_sentences))
        remainder = target_size % len(clean_sentences)

        logger.info(
            f"Generating ~{target_size} synthetic pairs "
            f"({passes} passes + {remainder} extra)"
        )

        for pass_num in range(passes):
            for clean in clean_sentences:
                noisy = self.noise_gen.apply_noise(clean)
                n_star = compute_noise_level(noisy, clean)
                noisy_texts.append(noisy)
                clean_texts.append(clean)
                noise_levels.append(n_star)

        # Fill remainder
        extra = self.rng.sample(clean_sentences, min(remainder, len(clean_sentences)))
        for clean in extra:
            noisy = self.noise_gen.apply_noise(clean)
            n_star = compute_noise_level(noisy, clean)
            noisy_texts.append(noisy)
            clean_texts.append(clean)
            noise_levels.append(n_star)

        logger.info(f"Generated {len(noisy_texts)} synthetic pairs")
        return noisy_texts, clean_texts, noise_levels

    # â”€â”€ Splitting â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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


# â”€â”€ Helper functions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _replace_mentions(text: str) -> str:
    """Replace @username patterns with @ANON."""
    return re.sub(r"@\w+", "@ANON", text)


def _replace_urls(text: str) -> str:
    """Replace URLs with <URL>."""
    url_pattern = r"https?://\S+|www\.\S+"
    return re.sub(url_pattern, "<URL>", text)





