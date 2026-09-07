# PyTorch Dataset for noisy-clean sentence pairs with byte-level encodings and noise labels (n*).


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

        # Tokenize input (noisy sentence) without fixed padding
        input_encoding = self.tokenizer(
            noisy,
            max_length=self.max_input_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize target (clean sentence) without fixed padding
        target_encoding = self.tokenizer(
            clean,
            max_length=self.max_target_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        # ByT5 convention: replace padding token IDs in labels with -100
        # so the cross-entropy loss ignores padding positions.
        labels = target_encoding["input_ids"].squeeze(0).clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_encoding["input_ids"].squeeze(0),
            "attention_mask": input_encoding["attention_mask"].squeeze(0),
            "labels": labels,
            "noise_level": torch.tensor(
                self.noise_levels[idx], dtype=torch.float32
            ),
        }


class NormalizationCollator:
    """
    Dynamic padding collator for variable-length ByT5 byte sequences.
    Pads input_ids with pad_token_id, attention_mask with 0, and labels with -100.
    Optionally right-pads to the nearest multiple of pad_to_multiple_of.
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        pad_to_multiple_of: Optional[int] = None,
    ):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [b["input_ids"] for b in batch]
        attention_mask = [b["attention_mask"] for b in batch]
        labels = [b["labels"] for b in batch]
        noise_level = torch.stack([b["noise_level"] for b in batch])

        padded_inputs = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        padded_masks = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )

        if self.pad_to_multiple_of is not None and self.pad_to_multiple_of > 0:
            seq_len = padded_inputs.shape[1]
            remainder = seq_len % self.pad_to_multiple_of
            if remainder > 0:
                pad_len = self.pad_to_multiple_of - remainder
                padded_inputs = torch.nn.functional.pad(
                    padded_inputs, (0, pad_len), value=self.pad_token_id
                )
                padded_masks = torch.nn.functional.pad(
                    padded_masks, (0, pad_len), value=0
                )
                padded_labels = torch.nn.functional.pad(
                    padded_labels, (0, pad_len), value=-100
                )

        return {
            "input_ids": padded_inputs,
            "attention_mask": padded_masks,
            "labels": padded_labels,
            "noise_level": noise_level,
        }


# Default collator instance using ByT5 standard pad_token_id=0
collate_fn = NormalizationCollator(pad_token_id=0)

