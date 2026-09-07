import pytest
import torch
from src.data.dataset import NormalizationDataset, NormalizationCollator


class DummyTokenizer:
    pad_token_id = 0

    def __call__(self, text, max_length=1024, padding=False, truncation=True, return_tensors="pt"):
        # Simple UTF-8 byte conversion like ByT5
        raw_bytes = list(text.encode("utf-8"))
        if truncation and len(raw_bytes) > max_length:
            raw_bytes = raw_bytes[:max_length]
        length = len(raw_bytes)
        if padding == "max_length":
            input_ids = raw_bytes + [self.pad_token_id] * (max_length - length)
            attention_mask = [1] * length + [0] * (max_length - length)
        else:
            input_ids = raw_bytes
            attention_mask = [1] * length
        return {
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
        }


@pytest.fixture
def tokenizer():
    return DummyTokenizer()


def test_dataset_uses_1024_as_ceiling_not_padding_length(tokenizer):
    dataset = NormalizationDataset(["maikli"], ["maikli"], tokenizer)
    item = dataset[0]
    assert item["input_ids"].numel() < 1024
    assert item["input_ids"].shape == (6,)


def test_collator_pads_only_to_batch_max(tokenizer):
    dataset = NormalizationDataset(
        ["short", "this is much longer text"],
        ["short", "this is much longer text"],
        tokenizer,
    )
    collator = NormalizationCollator(pad_token_id=tokenizer.pad_token_id)
    batch = collator([dataset[0], dataset[1]])

    expected_len = max(dataset[0]["input_ids"].numel(), dataset[1]["input_ids"].numel())
    assert batch["input_ids"].shape == (2, expected_len)
    assert batch["attention_mask"].shape == (2, expected_len)
    assert batch["labels"].shape == (2, expected_len)

    # First item was shorter, so index 0 has padding at the end
    short_len = dataset[0]["input_ids"].numel()
    assert torch.all(batch["input_ids"][0, short_len:] == tokenizer.pad_token_id)
    assert torch.all(batch["attention_mask"][0, short_len:] == 0)
    assert torch.all(batch["attention_mask"][0, :short_len] == 1)
    assert torch.all(batch["labels"][0, short_len:] == -100)


def test_collator_respects_pad_to_multiple_of(tokenizer):
    dataset = NormalizationDataset(
        ["1234567890"],  # length 10
        ["1234567890"],
        tokenizer,
    )
    collator = NormalizationCollator(
        pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=8,
    )
    batch = collator([dataset[0]])
    # 10 rounds up to 16
    assert batch["input_ids"].shape[1] == 16
    assert batch["attention_mask"].shape[1] == 16
    assert batch["labels"].shape[1] == 16
