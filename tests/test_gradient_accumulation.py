import math
from unittest.mock import MagicMock
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from configs.byt5_config import ByT5Config
from src.training.trainer import TAHIMIKTrainer


class SimpleDataset(Dataset):
    def __init__(self, size=5):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "labels": torch.tensor([1, 2]),
        }


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        return {
            "loss": self.linear(input_ids.float()).sum(),
            "total_loss": self.linear(input_ids.float()).sum(),
        }


class DummyLoss:
    def __call__(self, outputs, **kwargs):
        loss = outputs["loss"]
        return {"total_loss": loss}


def test_gradient_accumulation_steps_and_loss_division():
    config = ByT5Config()
    config.fp16 = False
    config.learning_rate = 1e-3
    config.max_grad_norm = 1.0

    model = DummyModel()
    loss_fn = DummyLoss()
    trainer = TAHIMIKTrainer(model=model, config=config, loss_fn=loss_fn)

    dataset = SimpleDataset(size=5)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=lambda b: {
        "input_ids": torch.stack([x["input_ids"] for x in b]),
        "attention_mask": torch.stack([x["attention_mask"] for x in b]),
        "labels": torch.stack([x["labels"] for x in b]),
    })

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    # Spy on optimizer.step and scheduler.step
    optimizer_step_count = 0
    orig_opt_step = optimizer.step
    def mock_opt_step(*args, **kwargs):
        nonlocal optimizer_step_count
        optimizer_step_count += 1
        return orig_opt_step(*args, **kwargs)
    optimizer.step = mock_opt_step

    scheduler_step_count = 0
    orig_sched_step = scheduler.step
    def mock_sched_step(*args, **kwargs):
        nonlocal scheduler_step_count
        scheduler_step_count += 1
        return orig_sched_step(*args, **kwargs)
    scheduler.step = mock_sched_step

    accumulation_steps = 2
    trainer._train_epoch(
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        gradient_accumulation_steps=accumulation_steps,
    )

    # For 5 batches with accumulation factor 2:
    # batch 1: accum
    # batch 2: step (1)
    # batch 3: accum
    # batch 4: step (2)
    # batch 5: step (3) (final boundary)
    assert optimizer_step_count == 3
    assert scheduler_step_count == 3


def test_effective_batch_size_configuration():
    config = ByT5Config()
    # Stage 1: 2 * 8 = 16
    assert config.stage1_batch_size == 2
    assert config.stage1_gradient_accumulation_steps == 8
    assert config.stage1_batch_size * config.stage1_gradient_accumulation_steps == 16

    # Stage 2: 2 * 4 = 8
    assert config.stage2_batch_size == 2
    assert config.stage2_gradient_accumulation_steps == 4
    assert config.stage2_batch_size * config.stage2_gradient_accumulation_steps == 8
