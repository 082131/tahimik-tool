# =============================================================================
# Configuration for the MrT5 baseline (fixed-rate compression).
#
# MrT5 applies one fixed deletion rate to every input, regardless of how
# noisy the sentence is. This is the second baseline, demonstrating the
# efficiency gain of compression but also its limitation when the deletion
# rate cannot adapt to per-sentence noise density.
#
# Reference: Kallini et al. (2025), "MrT5: Dynamic Token Merging for
#            Efficient Byte-Level Language Models" (ICLR 2025).
# =============================================================================

from dataclasses import dataclass
from configs.base import BaseConfig


@dataclass
class MrT5Config(BaseConfig):
    """MrT5 with fixed-rate compression — the efficiency baseline."""

    variant_name: str = "mrt5_fixed"

    use_compression: bool = True
    noise_adaptive: bool = False

    # ── Delete gate placement ───────────────────────────────────────────
    # MrT5 places the delete gate after an early encoder layer (layer 3
    # in the original paper). The layers before the gate produce
    # contextual representations; the gate then scores and removes bytes.
    delete_gate_layer: int = 3

    # ── Fixed deletion target ───────────────────────────────────────────
    # A single deletion ratio applied to every input. MrT5's PI controller
    # drives the actual deletion rate toward this target during training.
    fixed_deletion_target: float = 0.5

    # ── Gate regularizer (Equation 3 in MrT5 paper) ────────────────────
    # The gate regularizer encourages deletion by penalizing gate outputs
    # that are close to 0 (keep) rather than close to k (delete).
    gate_k: float = -30.0          # Large negative constant bounding the gate

    # ── Gumbel noise on the gate logits (MrT5 reference impl.) ──────────
    # The reference implementation adds Gumbel noise to the delete-gate
    # logits during training. It makes the keep/delete decision explorable:
    # without it the gate can commit early to a decision it then never
    # revisits, because nothing perturbs the score enough to try the
    # alternative. Training only — never applied at inference.
    use_gumbel_noise: bool = True

    # ── Rate loss weight ────────────────────────────────────────────────
    w_rate: float = 1.0

    # ── Attention regularizer ───────────────────────────────────────────
    # Prevents attention scores from inflating to circumvent the masking
    # effect of the delete gate (Appendix D of MrT5 paper).
    w_attn_reg: float = 0.01
