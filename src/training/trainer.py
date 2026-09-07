# Two-stage trainer supporting Stage 1 (synthetic pretraining) and Stage 2 (gold fine-tuning)
# with AdamW, cosine LR scheduling, and fp16 mixed precision.


import hashlib
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import get_scheduler

from src.data.dataset import NormalizationCollator, NormalizationDataset, collate_fn
from src.training.losses import TAHIMIKLoss
from src.utils.logging_utils import setup_logger
from src.utils.reproducibility import collect_run_metadata

logger = setup_logger("tahimik.trainer")


@dataclass
class HandoffRecord:
    source_checkpoint_path: str
    source_stage: str
    source_epoch: int
    source_val_loss: float
    target_stage: str
    restoration_timestamp: str
    restoration_success: bool
    model_fingerprint: str


def compute_model_fingerprint(model: nn.Module) -> str:
    """Computes a deterministic hex digest from the model's parameters."""
    hasher = hashlib.sha256()
    for name, param in sorted(model.state_dict().items()):
        hasher.update(name.encode("utf-8"))
        hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()[:16]


def compute_noise_band_diagnostics(
    noise_scores: torch.Tensor,
    deletion_rates: torch.Tensor,
) -> Dict[str, Any]:
    """
    Computes sample-weighted noise and deletion metrics across fixed noise bands:
    [0.0, 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 1.0].
    Empty bands return None for mean_deletion (never fabricated 0.0).
    """
    noise_scores = noise_scores.detach().cpu().view(-1)
    deletion_rates = deletion_rates.detach().cpu().view(-1)
    assert len(noise_scores) == len(deletion_rates), (
        f"Length mismatch: {len(noise_scores)} noise scores vs {len(deletion_rates)} deletion rates"
    )

    band_defs = [
        ("[0.0, 0.2)", 0.0, 0.2, False),
        ("[0.2, 0.4)", 0.2, 0.4, False),
        ("[0.4, 0.6)", 0.4, 0.6, False),
        ("[0.6, 0.8)", 0.6, 0.8, False),
        ("[0.8, 1.0]", 0.8, 1.0, True),
    ]

    bands_result = {}
    for name, low, high, inclusive_upper in band_defs:
        if inclusive_upper:
            mask = (noise_scores >= low) & (noise_scores <= high)
        else:
            mask = (noise_scores >= low) & (noise_scores < high)
        count = int(mask.sum().item())
        if count > 0:
            mean_del = float(deletion_rates[mask].mean().item())
        else:
            mean_del = None
        bands_result[name] = {
            "count": count,
            "mean_deletion": mean_del,
        }

    overall_mean_noise = float(noise_scores.mean().item()) if len(noise_scores) > 0 else 0.0
    overall_mean_deletion = float(deletion_rates.mean().item()) if len(deletion_rates) > 0 else 0.0

    return {
        "overall_mean_noise": overall_mean_noise,
        "overall_mean_deletion_rate": overall_mean_deletion,
        "bands": bands_result,
    }


class TAHIMIKTrainer:
    """
    Two-stage trainer for all TAHIMIK model variants.

    Handles the full training loop including:
        - AdamW optimizer with cosine LR scheduling
        - Precision controls (fp32 / fp16 / bf16) with AMP
        - Model gradient checkpointing
        - Gradient accumulation
        - Checkpoint saving and provenance
        - Training/validation logging per epoch
    """

    def __init__(self, model, config, loss_fn: TAHIMIKLoss):
        self.model = model
        self.config = config
        self.loss_fn = loss_fn

        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        # Precision and device validation
        precision = getattr(config, "precision", "fp16").lower()
        if self.device.type == "cpu" and precision in ("fp16", "bf16"):
            raise ValueError(
                f"Precision '{precision}' is not supported on CPU. Use 'fp32' on CPU."
            )
        if self.device.type == "cuda" and precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError(
                "bf16 precision requested but not supported on this CUDA device."
            )

        self.precision = precision
        if precision == "bf16":
            self.autocast_dtype = torch.bfloat16
            self.use_autocast = True
            self.use_scaler = False
        elif precision == "fp16":
            self.autocast_dtype = torch.float16
            self.use_autocast = (self.device.type == "cuda")
            self.use_scaler = (self.device.type == "cuda")
        else:  # fp32
            self.autocast_dtype = torch.float32
            self.use_autocast = False
            self.use_scaler = False

        self.fp16 = self.use_scaler
        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        self.scaler = GradScaler(device_type, enabled=self.use_scaler)

        # Gradient checkpointing
        if getattr(config, "gradient_checkpointing", False):
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            elif hasattr(self.model, "model") and hasattr(self.model.model, "gradient_checkpointing_enable"):
                self.model.model.gradient_checkpointing_enable()

            if hasattr(self.model, "config"):
                self.model.config.use_cache = False
            if hasattr(self.model, "model") and hasattr(self.model.model, "config"):
                self.model.model.config.use_cache = False

        # Best validation loss for checkpoint selection
        self.best_val_loss = float("inf")

        pad_id = getattr(getattr(self.model, "tokenizer", None), "pad_token_id", 0)
        self.collator = NormalizationCollator(pad_token_id=pad_id)


    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer per manuscript specification."""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
            weight_decay=self.config.weight_decay,
        )

    def _create_scheduler(self, optimizer, num_training_steps):
        """Create cosine LR scheduler with warmup."""
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        return get_scheduler(
            name=self.config.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    def _train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        gradient_accumulation_steps: int = 1,
    ) -> Dict[str, Any]:
        """Run one training epoch with gradient accumulation."""
        self.model.train()
        total_losses = {}
        num_batches = 0

        all_noise_scores = []
        all_deletion_rates = []
        last_coeff = None
        last_navg = None

        optimizer.zero_grad()
        num_items = len(dataloader)
        accum_steps = max(gradient_accumulation_steps, 1)

        for step, batch in enumerate(dataloader, start=1):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.use_autocast):
                model_outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    noise_level=batch.get("noise_level"),
                )

                losses = self.loss_fn(
                    model_outputs,
                    noise_level=batch.get("noise_level"),
                )

            # Divide loss by gradient accumulation steps before backward
            scaled_loss = losses["total_loss"] / accum_steps
            self.scaler.scale(scaled_loss).backward()

            # Step optimizer and scheduler at accumulation boundary or final batch
            if (step % accum_steps == 0) or (step == num_items):
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.scaler.step(optimizer)
                self.scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            # Collect diagnostics
            if "adaptive_coefficient" in model_outputs and model_outputs["adaptive_coefficient"] is not None:
                last_coeff = model_outputs["adaptive_coefficient"].detach().cpu()
            if "noise_average" in model_outputs and model_outputs["noise_average"] is not None:
                last_navg = model_outputs["noise_average"].detach().cpu()
            if "deletion_rate" in model_outputs and model_outputs["deletion_rate"] is not None:
                del_rates = model_outputs["deletion_rate"].detach().cpu().view(-1)
                n_scores = model_outputs.get("noise_scores")
                if n_scores is None:
                    n_scores = batch.get("noise_level")
                if n_scores is not None:
                    n_scores = n_scores.detach().cpu().view(-1)
                    if len(n_scores) == len(del_rates):
                        all_noise_scores.append(n_scores)
                        all_deletion_rates.append(del_rates)

            # Accumulate losses for logging
            for key, val in losses.items():
                if key not in total_losses:
                    total_losses[key] = 0.0
                total_losses[key] += val.item()
            num_batches += 1

        # Average over batches
        avg_losses: Dict[str, Any] = {
            k: v / max(num_batches, 1) for k, v in total_losses.items()
        }
        if all_noise_scores and all_deletion_rates:
            cat_noise = torch.cat(all_noise_scores)
            cat_del = torch.cat(all_deletion_rates)
            diag = compute_noise_band_diagnostics(cat_noise, cat_del)
            if last_coeff is not None:
                diag["adaptive_coefficient"] = float(last_coeff.item())
            if last_navg is not None:
                diag["noise_average"] = float(last_navg.item())
            avg_losses["diagnostics"] = diag

        return avg_losses

    @torch.no_grad()
    def _validate(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Run validation and return average losses."""
        self.model.eval()
        total_losses = {}
        num_batches = 0

        all_noise_scores = []
        all_deletion_rates = []
        last_coeff = None
        last_navg = None

        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.use_autocast):
                model_outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    noise_level=batch.get("noise_level"),
                )

                losses = self.loss_fn(
                    model_outputs,
                    noise_level=batch.get("noise_level"),
                )

            if "adaptive_coefficient" in model_outputs and model_outputs["adaptive_coefficient"] is not None:
                last_coeff = model_outputs["adaptive_coefficient"].detach().cpu()
            if "noise_average" in model_outputs and model_outputs["noise_average"] is not None:
                last_navg = model_outputs["noise_average"].detach().cpu()
            if "deletion_rate" in model_outputs and model_outputs["deletion_rate"] is not None:
                del_rates = model_outputs["deletion_rate"].detach().cpu().view(-1)
                n_scores = model_outputs.get("noise_scores")
                if n_scores is None:
                    n_scores = batch.get("noise_level")
                if n_scores is not None:
                    n_scores = n_scores.detach().cpu().view(-1)
                    if len(n_scores) == len(del_rates):
                        all_noise_scores.append(n_scores)
                        all_deletion_rates.append(del_rates)

            for key, val in losses.items():
                if key not in total_losses:
                    total_losses[key] = 0.0
                total_losses[key] += val.item()
            num_batches += 1

        avg_losses: Dict[str, Any] = {
            k: v / max(num_batches, 1) for k, v in total_losses.items()
        }
        if all_noise_scores and all_deletion_rates:
            cat_noise = torch.cat(all_noise_scores)
            cat_del = torch.cat(all_deletion_rates)
            diag = compute_noise_band_diagnostics(cat_noise, cat_del)
            if last_coeff is not None:
                diag["adaptive_coefficient"] = float(last_coeff.item())
            if last_navg is not None:
                diag["noise_average"] = float(last_navg.item())
            avg_losses["diagnostics"] = diag

        return avg_losses

    def _save_checkpoint(self, epoch: int, stage: str, val_loss: float):
        """Save a checkpoint if validation loss improved."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            checkpoint_dir = os.path.join(
                self.config.checkpoint_dir,
                self.config.variant_name,
            )
            os.makedirs(checkpoint_dir, exist_ok=True)

            path = os.path.join(
                checkpoint_dir, f"best_{stage}.pt"
            )
            provenance = collect_run_metadata(seed=getattr(self.config, "seed", 42))
            provenance["precision"] = self.precision
            provenance["gradient_checkpointing"] = getattr(self.config, "gradient_checkpointing", False)

            payload = {
                "schema_version": "1.0.0",
                "epoch": epoch,
                "stage": stage,
                "model_state_dict": self.model.state_dict(),
                "val_loss": float(val_loss),
                "model_fingerprint": compute_model_fingerprint(self.model),
                "config": asdict(self.config) if hasattr(self.config, "__dataclass_fields__") else {},
                "provenance": provenance,
            }
            torch.save(payload, path)
            logger.info(f"  Saved checkpoint: {path} (val_loss={val_loss:.4f})")

    def restore_best_stage1_for_handoff(self) -> HandoffRecord:
        """
        Restores best_stage1.pt into self.model before Stage 2 fine-tuning.
        Fails closed if the checkpoint is missing, corrupted, or has a stage mismatch.
        """
        checkpoint_dir = os.path.join(
            self.config.checkpoint_dir,
            self.config.variant_name,
        )
        stage1_path = os.path.join(checkpoint_dir, "best_stage1.pt")
        if not os.path.exists(stage1_path):
            raise FileNotFoundError(
                f"Cannot restore Stage 1 checkpoint: file not found at {stage1_path}. "
                "Stage 2 requires a valid best_stage1.pt from Stage 1."
            )

        checkpoint = torch.load(stage1_path, map_location=self.device)
        if checkpoint.get("stage") != "stage1":
            raise ValueError(
                f"Checkpoint at {stage1_path} stage mismatch: expected 'stage1', got '{checkpoint.get('stage')}'"
            )

        if "model_state_dict" not in checkpoint:
            raise ValueError(f"Malformed checkpoint at {stage1_path}: missing 'model_state_dict'")

        self.model.load_state_dict(checkpoint["model_state_dict"])
        fp = compute_model_fingerprint(self.model)

        record = HandoffRecord(
            source_checkpoint_path=stage1_path,
            source_stage="stage1",
            source_epoch=checkpoint.get("epoch", -1),
            source_val_loss=checkpoint.get("val_loss", float("inf")),
            target_stage="stage2",
            restoration_timestamp=datetime.now(timezone.utc).isoformat(),
            restoration_success=True,
            model_fingerprint=fp,
        )
        self.handoff_record = record
        logger.info(
            f"Successfully restored Stage 1 best checkpoint (epoch={record.source_epoch}, "
            f"val_loss={record.source_val_loss:.4f}) for Stage 2 handoff."
        )
        return record


    def train_stage(
        self,
        stage_name: str,
        train_dataset: NormalizationDataset,
        val_dataset: NormalizationDataset,
        epochs: int,
        batch_size: int,
        gradient_accumulation_steps: int = 1,
    ) -> Dict[str, list]:
        """
        Train a single stage (Stage 1 or Stage 2).

        Args:
            stage_name: 'stage1' (synthetic) or 'stage2' (gold).
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
            epochs: Number of epochs to train.
            batch_size: Physical batch size.
            gradient_accumulation_steps: Microbatches per optimizer step.

        Returns:
            History dict with per-epoch train and val losses.
        """
        accum_steps = max(gradient_accumulation_steps, 1)
        effective_batch_size = batch_size * accum_steps
        logger.info(f"{'='*60}")
        logger.info(
            f"Starting {stage_name}: {epochs} epochs, physical batch_size={batch_size}, "
            f"accum={accum_steps} (effective batch_size={effective_batch_size})"
        )
        logger.info(f"  Train samples: {len(train_dataset)}")
        logger.info(f"  Val samples:   {len(val_dataset)}")
        logger.info(f"{'='*60}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collator,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            collate_fn=self.collator,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )

        optimizer = self._create_optimizer()
        updates_per_epoch = math.ceil(len(train_loader) / accum_steps)
        num_training_steps = updates_per_epoch * epochs
        scheduler = self._create_scheduler(optimizer, num_training_steps)

        history = {"train": [], "val": []}

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # Train
            train_losses = self._train_epoch(
                train_loader,
                optimizer,
                scheduler,
                gradient_accumulation_steps=accum_steps,
            )
            # Validate
            val_losses = self._validate(val_loader)

            epoch_time = time.time() - epoch_start

            history["train"].append(train_losses)
            history["val"].append(val_losses)

            # Log
            logger.info(
                f"  [{stage_name}] Epoch {epoch}/{epochs} "
                f"({epoch_time:.1f}s) - "
                f"train_loss={train_losses['total_loss']:.4f} "
                f"val_loss={val_losses['total_loss']:.4f}"
            )

            # Log component losses if present
            for key in ["l_ce", "l_rate", "l_attn_reg", "l_ne"]:
                if key in train_losses and key in val_losses:
                    logger.info(
                        f"    {key}: train={train_losses[key]:.4f} "
                        f"val={val_losses[key]:.4f}"
                    )

            if "diagnostics" in train_losses:
                diag = train_losses["diagnostics"]
                logger.info(
                    f"    Train Diagnostics: mean_noise={diag['overall_mean_noise']:.4f}, "
                    f"mean_del={diag['overall_mean_deletion_rate']:.4f}"
                )
                band_info = []
                for b, d in diag["bands"].items():
                    val_str = f"{d['mean_deletion']:.3f}" if d["mean_deletion"] is not None else "N/A"
                    band_info.append(f"{b}: {val_str} (n={d['count']})")
                logger.info(f"    Train Noise Bands: {', '.join(band_info)}")

            # Checkpoint
            self._save_checkpoint(epoch, stage_name, val_losses["total_loss"])

        return history

    def train(
        self,
        stage1_train: Optional[NormalizationDataset] = None,
        stage1_val: Optional[NormalizationDataset] = None,
        stage2_train: Optional[NormalizationDataset] = None,
        stage2_val: Optional[NormalizationDataset] = None,
    ) -> Dict[str, Dict]:
        """
        Run the full two-stage training pipeline.

        Stage 1 is skipped if stage1_train/stage1_val are None (useful
        for debugging with gold data only).

        Args:
            stage1_train, stage1_val: Synthetic pretraining datasets.
            stage2_train, stage2_val: Gold standard fine-tuning datasets.

        Returns:
            Dict with 'stage1' and 'stage2' history dicts.
        """
        full_history = {}

        # --- Stage 1: Synthetic Pretraining ----------------------------------
        if stage1_train is not None and stage1_val is not None:
            self.best_val_loss = float("inf")
            stage1_accum = getattr(self.config, "stage1_gradient_accumulation_steps", 1)
            stage1_history = self.train_stage(
                stage_name="stage1",
                train_dataset=stage1_train,
                val_dataset=stage1_val,
                epochs=self.config.stage1_epochs,
                batch_size=self.config.stage1_batch_size,
                gradient_accumulation_steps=stage1_accum,
            )
            full_history["stage1"] = stage1_history
        else:
            logger.info("Skipping Stage 1 (no synthetic data provided)")

        # --- Stage 2: Gold Standard Fine-tuning ------------------------------
        if stage2_train is not None and stage2_val is not None:
            # If Stage 1 ran, restore best_stage1.pt before initializing Stage 2
            if stage1_train is not None and stage1_val is not None:
                handoff = self.restore_best_stage1_for_handoff()
                full_history["handoff"] = asdict(handoff)

            self.best_val_loss = float("inf")
            stage2_accum = getattr(self.config, "stage2_gradient_accumulation_steps", 1)
            stage2_history = self.train_stage(
                stage_name="stage2",
                train_dataset=stage2_train,
                val_dataset=stage2_val,
                epochs=self.config.stage2_epochs,
                batch_size=self.config.stage2_batch_size,
                gradient_accumulation_steps=stage2_accum,
            )
            full_history["stage2"] = stage2_history

        else:
            logger.info("Skipping Stage 2 (no gold standard data provided)")

        logger.info("Training complete.")
        return full_history

    def load_checkpoint(self, checkpoint_path: str):
        """Load a saved checkpoint into the model."""
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(
            f"Loaded checkpoint from {checkpoint_path} "
            f"(epoch={checkpoint['epoch']}, "
            f"stage={checkpoint['stage']}, "
            f"val_loss={checkpoint['val_loss']:.4f})"
        )


# Alias for compatibility
Trainer = TAHIMIKTrainer


