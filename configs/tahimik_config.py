# =============================================================================
# Configuration for the proposed Noise-Adaptive ByT5 model (TAHIMIK).
#
# This is the study's main contribution. It extends MrT5's delete gate
# with a learned noise estimator that conditions the deletion rate on each
# input's noise characteristics. Noisier sentences receive less compression
# (preserving the bytes needed for correction), while cleaner sentences
# receive more compression (improving efficiency).
#
# Key formula (from the manuscript's Noise-Adaptive Deletion section):
#   keep_score_shift = cn * (n - navg)
# where:
#   cn   = learned coefficient
#   n    = predicted noise score for this sentence [0, 1]
#   navg = running average of noise scores across the training set
# =============================================================================

from dataclasses import dataclass
from configs.base import BaseConfig


@dataclass
class TAHIMIKConfig(BaseConfig):
    """Noise-Adaptive ByT5 — the proposed TAHIMIK model."""

    variant_name: str = "tahimik_noise_adaptive"

    use_compression: bool = True
    noise_adaptive: bool = True

    # ── Delete gate placement (same as MrT5 for fair comparison) ────────
    delete_gate_layer: int = 3

    # ── Gate constants ──────────────────────────────────────────────────
    gate_k: float = -30.0

    # ── Noise-adaptive deletion target ──────────────────────────────────
    # d_target(x_i) = d_max * (1 - n_i)
    # Clean inputs (n≈0) → target ≈ d_max (aggressive compression)
    # Noisy inputs (n≈1) → target ≈ 0 (minimal compression)
    d_max: float = 0.5             # Maximum deletion fraction

    # ── Noise estimator architecture ────────────────────────────────────
    # The noise estimator is a small feed-forward network that maps the
    # mean-pooled encoder hidden states to a single noise score via sigmoid.
    noise_estimator_hidden_dim: int = 256

    # ── Running average momentum for navg ───────────────────────────────
    # navg tracks the moving average of noise scores. It enters the gate's
    # conditioning term cn*(n - navg) but is detached from gradients so
    # the estimator only learns from L_NE.
    noise_avg_momentum: float = 0.99

    # ── Gumbel noise on the gate logits (MrT5 reference impl.) ──────────
    # The reference implementation adds Gumbel noise to the delete-gate
    # logits during training. It makes the keep/delete decision explorable:
    # without it the gate can commit early to a decision it then never
    # revisits, because nothing perturbs the score enough to try the
    # alternative. Training only — never applied at inference.
    use_gumbel_noise: bool = True

    # ── Loss weights (from the Loss Computation section) ────────────────
    # L = L_CE + w_rate * L_rate + w_attn_reg * L_attn_reg + L_NE
    w_rate: float = 1.0            # Per-sentence rate loss weight
    w_attn_reg: float = 0.01       # Attention regularizer weight
    # L_NE has implicit weight 1.0 (noise estimation loss)

    # ── Noise score supervision ─────────────────────────────────────────
    # n* is computed as byte-level edit distance between noisy and clean
    # sentences divided by the length of the longer sentence, producing
    # a value in [0, 1]. This is the supervision target for the noise
    # estimator during training.
    noise_label_method: str = "edit_distance_ratio"
