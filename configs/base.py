# =============================================================================
# Base configuration shared across all three TAHIMIK model variants.
#
# All hyperparameters that must remain constant across variants (control
# variables from the conceptual framework) are defined here: data splits,
# max sequence length, optimizer settings, and evaluation parameters.
# =============================================================================

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BaseConfig:
    """
    Shared configuration for ByT5, MrT5, and Noise-Adaptive ByT5.

    Control variables (held constant per the thesis conceptual framework):
        - Training data and partitioning
        - Model hyperparameters (base ByT5 variant, optimizer, LR)
        - Hardware and evaluation settings
    """

    # ── Model identity ──────────────────────────────────────────────────
    model_name: str = "google/byt5-base"   # HuggingFace model ID (manuscript specification)


    # ── Byte-level sequence constraints ─────────────────────────────────
    # ByT5 processes raw UTF-8 bytes. The manuscript fixes the maximum
    # input length at 1024 bytes (~170 words of plain text).
    max_input_length: int = 1024
    max_target_length: int = 1024

    # ── Data partitioning (Table 1 in the manuscript) ───────────────────
    # Gold standard: 80/10/10 train/val/test
    gold_train_ratio: float = 0.80
    gold_val_ratio: float = 0.10
    gold_test_ratio: float = 0.10

    # Synthetic: 90/10 train/val (no test — final eval uses gold test set)
    synthetic_train_ratio: float = 0.90
    synthetic_val_ratio: float = 0.10

    # ── Optimizer (AdamW, as described in the Backpropagation section) ───
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # ── Training schedule ───────────────────────────────────────────────
    # Stage 1: synthetic data pretraining
    # Physical batch size = 2, accumulation = 8 -> effective batch size = 16
    stage1_epochs: int = 3
    stage1_batch_size: int = 2
    stage1_gradient_accumulation_steps: int = 8

    # Stage 2: gold standard fine-tuning
    # Physical batch size = 2, accumulation = 4 -> effective batch size = 8
    stage2_epochs: int = 10
    stage2_batch_size: int = 2
    stage2_gradient_accumulation_steps: int = 4

    warmup_ratio: float = 0.06
    lr_scheduler_type: str = "cosine"

    def __post_init__(self):
        if self.stage1_batch_size <= 0 or self.stage1_gradient_accumulation_steps <= 0:
            raise ValueError("Stage 1 batch size and accumulation steps must be positive integers.")
        if self.stage2_batch_size <= 0 or self.stage2_gradient_accumulation_steps <= 0:
            raise ValueError("Stage 2 batch size and accumulation steps must be positive integers.")

    # ── Evaluation ──────────────────────────────────────────────────────
    eval_batch_size: int = 16
    num_beams: int = 4             # Beam search during inference
    inference_runs: int = 20       # Repeated runs for GPU memory measurement
    warmup_passes: int = 5         # CUDA warm-up before timing

    # ── Reproducibility ─────────────────────────────────────────────────
    seed: int = 42

    # ── Paths (override per environment) ────────────────────────────────
    data_dir: str = "data"
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"

    # ── Device ──────────────────────────────────────────────────────────
    device: str = "cuda"           # "cuda" for Colab A100, "cpu" for local
    fp16: bool = True              # Mixed precision for efficiency
