# =============================================================================
# Loss Functions for TAHIMIK Training
#
# The total loss is a weighted sum of four components (from the manuscript's
# Loss Computation section):
#
#   L = L_CE + w_rate * L_rate + w_attn_reg * L_attn_reg + L_NE
#
# Components:
#   L_CE       — Cross-entropy between predicted and target byte sequences.
#               Standard seq2seq loss; already computed by the model.
#
#   L_rate     — Per-sentence deletion rate loss. Penalizes deviation of the
#               actual deletion rate from the target:
#               - MrT5: target = fixed_deletion_target (e.g. 0.5)
#               - TAHIMIK: target = d_max * (1 - n_i), noise-adaptive
#
#   L_attn_reg — Attention regularizer. Prevents attention scores from
#               inflating to circumvent the soft masking of the delete gate
#               (MrT5 Appendix D). Penalizes large attention weights on
#               positions that the gate marked for deletion.
#
#   L_NE       — Noise estimator loss. MSE between predicted noise score n
#               and ground-truth n* (byte-level edit distance ratio).
#               Only applies to TAHIMIK. This is the ONLY signal that trains
#               the noise estimator — n is detached everywhere else.
# =============================================================================

import torch
import torch.nn as nn
from typing import Dict, Optional


class TAHIMIKLoss(nn.Module):
    """
    Combined loss function for all three TAHIMIK model variants.

    For ByT5 baseline: only L_CE is active.
    For MrT5: L_CE + L_rate + L_attn_reg.
    For TAHIMIK: L_CE + L_rate + L_attn_reg + L_NE.

    Args:
        w_rate: Weight for the deletion rate loss.
        w_attn_reg: Weight for the attention regularizer.
        use_compression: Whether the model uses a delete gate.
        noise_adaptive: Whether the model uses noise-adaptive deletion.
    """

    def __init__(
        self,
        w_rate: float = 1.0,
        w_attn_reg: float = 0.01,
        use_compression: bool = True,
        noise_adaptive: bool = False,
    ):
        super().__init__()

        self.w_rate = w_rate
        self.w_attn_reg = w_attn_reg
        self.use_compression = use_compression
        self.noise_adaptive = noise_adaptive

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        noise_level: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the combined loss from model outputs.

        Args:
            model_outputs: Dict from the model's forward pass containing
                'loss' (L_CE), and optionally 'gate_outputs', 'deletion_rate',
                'target_deletion_rate', 'noise_scores', 'ne_loss'.
            noise_level: Ground-truth n* values. Shape: (batch,).
                Required for TAHIMIK's L_NE.

        Returns:
            Dict with 'total_loss' and individual component losses for logging.
        """
        losses = {}

        # ── L_CE: Cross-entropy (always present) ───────────────────────
        l_ce = model_outputs["loss"]
        losses["l_ce"] = l_ce
        total = l_ce

        if self.use_compression:
            # ── L_rate: Deletion rate loss ──────────────────────────────
            deletion_rate = model_outputs["deletion_rate"]

            if self.noise_adaptive:
                # TAHIMIK: target is noise-adaptive
                target_rate = model_outputs["target_deletion_rate"]
            else:
                # MrT5: fixed target for all sentences
                target_rate = torch.full_like(
                    deletion_rate,
                    model_outputs.get("fixed_deletion_target", 0.5),
                )

            l_rate = ((deletion_rate - target_rate) ** 2).mean()
            losses["l_rate"] = l_rate
            total = total + self.w_rate * l_rate

            # ── L_attn_reg: Gate commitment regularizer ─────────────────
            # Encourages the gate to commit to keep (p→1) or delete (p→0)
            # rather than sitting at an indecisive p≈0.5, which would leave
            # every byte half-attended and blur the compression decision.
            #
            #   L = mean( 4 * p * (1 - p) )
            #
            # Maximal (1.0) at p=0.5, zero at p=0 and p=1, so it is SYMMETRIC:
            # it expresses no preference between keeping and deleting. Which
            # way a byte goes is decided by L_rate and L_CE.
            #
            # NOTE (deviation from the manuscript): MrT5 Appendix D defines
            # this term over attention weights, penalising attention paid to
            # deleted positions, which requires materialising per-head
            # attention matrices. The symmetric form below serves the same
            # purpose — stopping the gate from hedging — at negligible cost.
            # The previous implementation penalised (1 - gate/k)^2, which is
            # minimised by sending EVERY gate to k, i.e. deleting the whole
            # sequence; it was the only gradient reaching the gate.
            keep_prob = model_outputs["keep_prob"]

            l_attn_reg = (4.0 * keep_prob * (1.0 - keep_prob)).mean()
            losses["l_attn_reg"] = l_attn_reg
            total = total + self.w_attn_reg * l_attn_reg

        if self.noise_adaptive:
            # ── L_NE: Noise estimator loss ──────────────────────────────
            if "ne_loss" in model_outputs:
                l_ne = model_outputs["ne_loss"]
            elif noise_level is not None and "noise_scores" in model_outputs:
                l_ne = nn.functional.mse_loss(
                    model_outputs["noise_scores"], noise_level
                )
            else:
                l_ne = torch.tensor(0.0, device=l_ce.device)

            losses["l_ne"] = l_ne
            total = total + l_ne

        losses["total_loss"] = total
        return losses
