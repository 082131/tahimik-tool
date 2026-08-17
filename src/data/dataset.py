# =============================================================================
# PyTorch Dataset for TAHIMIK Text Normalization
#
# Handles both synthetic pretraining data (Stage 1) and gold standard
# fine-tuning data (Stage 2). Each sample contains:
#   - noisy_text: the input social media sentence
#   - clean_text: the reference normalized output
#   - noise_level: n* (byte-level edit distance ratio), supervision for
#                  the noise estimator
#   - input_ids / attention_mask: ByT5-compatible byte-level tensors
#   - labels: target byte IDs for the decoder
#
# The same Dataset class is used for all three model variants to ensure
# the control variable (training data) is held constant.
# =============================================================================

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional
from transformers import AutoTokenizer

from src.data.noise_label import compute_noise_level


class NormalizationDataset(Dataset):
    """
    Dataset for Tagalog/Taglish text normalization training and evaluation.

    Each item provides the noisy input, clean target, and precomputed
    noise level n* for the noise estimator's supervision signal.

    Args:
        noisy_texts: List of noisy input sentences.
        clean_texts: List of corresponding normalized reference sentences.
        tokenizer: HuggingFace ByT5 tokenizer for byte-level encoding.
        max_input_length: Maximum input byte sequence length (default 1024).
        max_target_length: Maximum target byte sequence length (default 1024).
        precomputed_noise_levels: Optional precomputed n* values. If None,
            they are computed on-the-fly from the text pairs.
    """

    def __init__(
        self,
        noisy_texts: List[str],
        clean_texts: List[str],
        tokenizer: AutoTokenizer,
        max_input_length: int = 1024,
        max_target_length: int = 1024,
        precomputed_noise_levels: Optional[List[float]] = None,
    ):
        assert len(noisy_texts) == len(clean_texts), (
            f"Mismatched pair count: {len(noisy_texts)} noisy vs {len(clean_texts)} clean"
        )

        self.noisy_texts = noisy_texts
        self.clean_texts = clean_texts
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        # Precompute noise levels if not provided — this avoids redundant
        # edit distance computation on every epoch.
        if precomputed_noise_levels is not None:
            self.noise_levels = precomputed_noise_levels
        else:
            self.noise_levels = [
                compute_noise_level(noisy, clean)
                for noisy, clean in zip(noisy_texts, clean_texts)
            ]

    def __len__(self) -> int:
        return len(self.noisy_texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        noisy = self.noisy_texts[idx]
        clean = self.clean_texts[idx]

        # Tokenize input (noisy sentence)
        input_encoding = self.tokenizer(
            noisy,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize target (clean sentence)
        target_encoding = self.tokenizer(
            clean,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # ByT5 convention: replace padding token IDs in labels with -100
        # so the cross-entropy loss ignores padding positions.
        labels = target_encoding["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": labels,
            "noise_level": torch.tensor(
                self.noise_levels[idx], dtype=torch.float32
            ),
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for DataLoader.

    Stacks individual samples into batched tensors. The noise_level
    field is gathered into a 1D tensor for batch-level loss computation.
    """
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "noise_level": torch.stack([b["noise_level"] for b in batch]),
    }
