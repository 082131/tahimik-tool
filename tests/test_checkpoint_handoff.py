import os
from dataclasses import dataclass
import pytest
import torch
import torch.nn as nn

from src.training.trainer import (
    TAHIMIKTrainer,
    HandoffRecord,
    compute_model_fingerprint,
)
from src.training.losses import TAHIMIKLoss


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.tensor([0.0]))

    def forward(self, *args, **kwargs):
        return {"loss": torch.tensor(0.5), "logits": self.param}


@dataclass
class TinyConfig:
    checkpoint_dir: str
    variant_name: str = "test_variant"
    device: str = "cpu"
    fp16: bool = False
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.0
    lr_scheduler_type: str = "linear"
    eval_batch_size: int = 1


def test_handoff_restores_best_epoch_not_final_epoch(tmp_path):
    config = TinyConfig(checkpoint_dir=str(tmp_path))
    model = TinyModel()
    loss_fn = TAHIMIKLoss(use_compression=False)
    trainer = TAHIMIKTrainer(model, config, loss_fn)

    # Simulate 3 epochs of Stage 1:
    # Epoch 1: val_loss 1.0 (weight = 1.0) -> best
    model.param.data.fill_(1.0)
    trainer._save_checkpoint(epoch=1, stage="stage1", val_loss=1.0)

    # Epoch 2: val_loss 0.5 (weight = 2.0) -> best
    model.param.data.fill_(2.0)
    trainer._save_checkpoint(epoch=2, stage="stage1", val_loss=0.5)

    # Epoch 3: val_loss 1.8 (weight = 3.0) -> worse, not saved as best
    model.param.data.fill_(3.0)
    trainer._save_checkpoint(epoch=3, stage="stage1", val_loss=1.8)

    # Current model weight is 3.0 (from epoch 3)
    assert model.param.item() == 3.0

    # Execute handoff
    record = trainer.restore_best_stage1_for_handoff()

    assert isinstance(record, HandoffRecord)
    assert record.source_epoch == 2
    assert record.source_val_loss == pytest.approx(0.5)
    assert record.restoration_success is True
    # Weight was restored to epoch 2 (2.0)
    assert model.param.item() == pytest.approx(2.0)


def test_handoff_fails_closed_when_checkpoint_missing(tmp_path):
    config = TinyConfig(checkpoint_dir=str(tmp_path))
    model = TinyModel()
    loss_fn = TAHIMIKLoss(use_compression=False)
    trainer = TAHIMIKTrainer(model, config, loss_fn)

    with pytest.raises(FileNotFoundError, match="Cannot restore Stage 1 checkpoint"):
        trainer.restore_best_stage1_for_handoff()


def test_handoff_fails_closed_on_stage_mismatch(tmp_path):
    config = TinyConfig(checkpoint_dir=str(tmp_path))
    model = TinyModel()
    loss_fn = TAHIMIKLoss(use_compression=False)
    trainer = TAHIMIKTrainer(model, config, loss_fn)

    # Save a checkpoint with stage='stage2' at best_stage1.pt path
    os.makedirs(os.path.join(str(tmp_path), config.variant_name), exist_ok=True)
    bad_path = os.path.join(str(tmp_path), config.variant_name, "best_stage1.pt")
    torch.save({"stage": "stage2", "epoch": 1, "model_state_dict": model.state_dict()}, bad_path)

    with pytest.raises(ValueError, match="stage mismatch"):
        trainer.restore_best_stage1_for_handoff()
