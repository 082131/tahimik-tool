import pytest
import torch
import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration

from configs.byt5_config import ByT5Config
from src.training.trainer import (
    TAHIMIKTrainer,
    validate_checkpoint_architecture,
)


class MockModel(nn.Module):
    def __init__(self, d_model=1536, num_layers=18, num_decoder_layers=6, vocab_size=384):
        super().__init__()
        self.config = T5Config(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_decoder_layers=num_decoder_layers,
            decoder_start_token_id=0,
        )
        self.linear = nn.Linear(d_model, 2)


def test_base_accepts_matching_base_checkpoint():
    model = MockModel(d_model=1536, num_layers=18, num_decoder_layers=6)
    checkpoint = {
        "schema_version": "1.0.0",
        "stage": "stage1",
        "architecture": {
            "model_name": "google/byt5-base",
            "d_model": 1536,
            "num_encoder_layers": 18,
            "num_decoder_layers": 6,
            "vocab_size": 384,
        },
        "model_state_dict": model.state_dict(),
    }
    # Must not raise
    validate_checkpoint_architecture(checkpoint, model)


def test_base_rejects_small_checkpoint_with_all_mismatches_listed():
    model = MockModel(d_model=1536, num_layers=18, num_decoder_layers=6)
    small_checkpoint = {
        "schema_version": "1.0.0",
        "stage": "stage1",
        "architecture": {
            "model_name": "google/byt5-small",
            "d_model": 1472,
            "num_encoder_layers": 12,
            "num_decoder_layers": 4,
            "vocab_size": 384,
        },
        "model_state_dict": {},
    }
    with pytest.raises(ValueError) as exc_info:
        validate_checkpoint_architecture(small_checkpoint, model)

    err_msg = str(exc_info.value)
    assert "d_model" in err_msg
    assert "1472" in err_msg and "1536" in err_msg
    assert "num_encoder_layers" in err_msg
    assert "12" in err_msg and "18" in err_msg
    assert "num_decoder_layers" in err_msg
    assert "4" in err_msg and "6" in err_msg


def test_legacy_checkpoint_fails_closed_unless_allow_legacy_specified():
    model = MockModel(d_model=1536, num_layers=18, num_decoder_layers=6)
    legacy_checkpoint = {
        "schema_version": "1.0.0",
        "stage": "stage1",
        "model_state_dict": model.state_dict(),
    }
    with pytest.raises(ValueError, match="lacks architecture metadata"):
        validate_checkpoint_architecture(legacy_checkpoint, model, allow_legacy=False)

    # Allowed with flag
    validate_checkpoint_architecture(legacy_checkpoint, model, allow_legacy=True)
