import pytest
import torch
import torch.nn as nn

from configs.base import BaseConfig
from configs.byt5_config import ByT5Config
from configs.mrt5_config import MrT5Config
from configs.tahimik_config import TAHIMIKConfig
from src.training.trainer import TAHIMIKTrainer


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)
        self.gradient_checkpointing_enabled = False

    def gradient_checkpointing_enable(self, **kwargs):
        self.gradient_checkpointing_enabled = True

    def forward(self, *args, **kwargs):
        return {"loss": torch.tensor(0.5)}


def test_all_variants_inherit_identical_memory_settings():
    configs = [ByT5Config(), MrT5Config(), TAHIMIKConfig()]
    checkpointings = {c.gradient_checkpointing for c in configs}
    precisions = {c.precision for c in configs}

    assert len(checkpointings) == 1, f"Mismatched checkpointing: {checkpointings}"
    assert checkpointings == {True}
    assert len(precisions) == 1, f"Mismatched precision: {precisions}"
    assert precisions == {"fp16"}


def test_unsupported_precision_value_rejected():
    with pytest.raises(ValueError, match="Unsupported precision"):
        BaseConfig(precision="int8")


def test_cpu_with_fp16_or_bf16_fails_before_training():
    config = ByT5Config()
    config.device = "cpu"
    config.precision = "fp16"

    model = DummyModel()
    with pytest.raises(ValueError, match="not supported on CPU"):
        TAHIMIKTrainer(model=model, config=config, loss_fn=lambda *a, **k: {})


def test_trainer_enables_gradient_checkpointing():
    config = ByT5Config()
    config.device = "cpu"
    config.precision = "fp32"
    config.gradient_checkpointing = True

    model = DummyModel()
    trainer = TAHIMIKTrainer(model=model, config=config, loss_fn=lambda *a, **k: {})
    assert model.gradient_checkpointing_enabled is True
