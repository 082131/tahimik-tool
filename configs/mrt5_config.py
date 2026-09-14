# Configuration for fixed-rate byte compression (MrT5).

from dataclasses import dataclass
from configs.base import BaseConfig


@dataclass
class MrT5Config(BaseConfig):
    """MrT5 fixed-rate compression configuration."""

    variant_name: str = "mrt5_fixed"

    use_compression: bool = True
    noise_adaptive: bool = False

    # ── Delete gate placement ───────────────────────────────────────────
    # MrT5 places the delete gate after an early encoder layer (layer 3
    # in the original paper). The layers before the gate produce
    # contextual representations; the gate then scores and removes bytes.
    delete_gate_layer: int = 2

    # ── Fixed deletion target ───────────────────────────────────────────
    # A single deletion ratio applied to every input. MrT5's PI controller
    # drives the actual deletion rate toward this target during training.
    fixed_deletion_target: float = 0.5

    # ── Gate regularizer (Equation 3 in MrT5 paper) ────────────────────
    # The gate regularizer encourages deletion by penalizing gate outputs
    # that are close to 0 (keep) rather than close to k (delete).
    gate_k: float = -10.0          # Stanford MrT5 sigmoid_mask_scale

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
