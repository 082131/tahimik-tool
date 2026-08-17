# =============================================================================
# Delete Gate — Byte-Level Compression with Noise-Adaptive Conditioning
#
# The delete gate assigns every byte position a keep/delete score and
# removes low-scoring bytes from the sequence. It operates in two modes:
#
#   TRAINING (soft deletion):
#     Gate outputs are applied as soft attention masks. Low-scoring bytes
#     are masked out of subsequent attention computations, but the sequence
#     length is not physically reduced. This keeps the gradient flow
#     fully differentiable.
#
#   INFERENCE (hard deletion):
#     Bytes with gate scores below the threshold (k/2) are physically
#     removed from the hidden state sequence, producing a shorter input
#     for the remaining encoder blocks. This is where the computational
#     savings occur.
#
# NOISE-ADAPTIVE CONDITIONING (TAHIMIK's contribution):
#   The vanilla MrT5 gate applies a fixed deletion rate. TAHIMIK extends
#   this by shifting every keep score based on the predicted noise level:
#
#       keep_score_shift = cn * (n - navg)
#
#   where cn is a learned coefficient, n is the noise score from the
#   noise estimator, and navg is the running average of noise scores.
#   A noisier sentence (n > navg) shifts scores UP → fewer deletions.
#   A cleaner sentence (n < navg) shifts scores DOWN → more deletions.
#
# Reference: MrT5 (Kallini et al., 2025), Equations 1-3, adapted with
#            noise-adaptive conditioning per the TAHIMIK manuscript.
# =============================================================================

import torch
import torch.nn as nn
from typing import Tuple, Optional


class DeleteGate(nn.Module):
    """
    Learned delete gate for byte-level sequence compression.

    Supports both fixed-rate (MrT5) and noise-adaptive (TAHIMIK) modes.

    Args:
        hidden_dim: Dimensionality of encoder hidden states (d_model).
        k: Large negative constant bounding the gate output range.
            Gate outputs are bounded in [k, 0]. Default: -30.0.
        noise_adaptive: If True, use noise-adaptive conditioning.
            If False, behave as a standard MrT5 fixed-rate gate.
        noise_avg_momentum: Exponential moving average momentum for navg.
    """

    def __init__(
        self,
        hidden_dim: int,
        k: float = -30.0,
        noise_adaptive: bool = True,
        noise_avg_momentum: float = 0.99,
    ):
        super().__init__()

        self.k = k
        self.noise_adaptive = noise_adaptive
        self.noise_avg_momentum = noise_avg_momentum

        # ── Gate scoring layers (Equation 1 from MrT5) ─────────────────
        # G = k * sigmoid(LayerNorm(H) @ W + b)
        # This produces per-byte scores in [k, 0].
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.gate_linear = nn.Linear(hidden_dim, 1)

        # ── Noise-adaptive conditioning ─────────────────────────────────
        # cn is a learned scalar coefficient that controls how strongly
        # the noise score shifts the keep/delete decision.
        if noise_adaptive:
            self.cn = nn.Parameter(torch.tensor(1.0))

            # Running average of noise scores (not a learnable parameter —
            # updated via EMA during training, frozen during inference)
            self.register_buffer("noise_avg", torch.tensor(0.5))

        # Hard deletion threshold: midpoint of the gate range [k, 0]
        self.hard_threshold = k / 2.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        noise_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute gate scores and apply soft or hard deletion.

        Args:
            hidden_states: Encoder hidden states at the gate layer.
                Shape: (batch_size, seq_len, hidden_dim)
            attention_mask: Binary mask (1=real, 0=pad).
                Shape: (batch_size, seq_len)
            noise_scores: Per-sentence noise scores from the noise estimator.
                Shape: (batch_size,). Required if noise_adaptive=True.

        Returns:
            gate_outputs: Per-byte gate scores in [k, 0].
                Shape: (batch_size, seq_len, 1)
            kept_mask: Binary mask of bytes that survived deletion.
                Shape: (batch_size, seq_len)
            deletion_rate: Actual fraction of bytes deleted per sentence.
                Shape: (batch_size,)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # ── Step 1: Compute raw gate scores ─────────────────────────────
        # LayerNorm → Linear → rescaled sigmoid in [k, 0]
        normed = self.layer_norm(hidden_states)
        logits = self.gate_linear(normed)  # (batch, seq, 1)

        # Rescaled sigmoid: k * sigmoid(logits), bounded in [k, 0]
        gate_outputs = self.k * torch.sigmoid(logits)

        # ── Step 2: Apply noise-adaptive shift ──────────────────────────
        if self.noise_adaptive and noise_scores is not None:
            # Detach noise scores so gate gradients don't flow back
            # to the noise estimator (manuscript: "n is detached here")
            n_detached = noise_scores.detach()

            # Update running average during training
            if self.training:
                batch_avg = n_detached.mean()
                self.noise_avg = (
                    self.noise_avg_momentum * self.noise_avg
                    + (1 - self.noise_avg_momentum) * batch_avg
                )

            # Shift = cn * (n - navg)
            # Shape: (batch_size,) → (batch_size, 1, 1) for broadcasting
            shift = self.cn * (n_detached - self.noise_avg)
            shift = shift.unsqueeze(1).unsqueeze(2)

            # Positive shift (noisy) → scores closer to 0 → keep more
            # Negative shift (clean) → scores closer to k → delete more
            gate_outputs = gate_outputs + shift

            # Clamp back to valid range [k, 0]
            gate_outputs = gate_outputs.clamp(min=self.k, max=0.0)

        # ── Step 3: Apply deletion ──────────────────────────────────────
        if self.training:
            # Soft deletion: gate outputs become attention masks
            # (handled externally by the model's attention modification)
            kept_mask = torch.ones(batch_size, seq_len, device=hidden_states.device)
        else:
            # Hard deletion: bytes below threshold are removed
            kept_mask = (gate_outputs.squeeze(-1) > self.hard_threshold).float()

            # Never delete padding positions (they're already masked)
            kept_mask = kept_mask * attention_mask.float()

        # ── Step 4: Compute deletion rate ───────────────────────────────
        # Count real tokens (non-padding) and how many were kept
        real_tokens = attention_mask.float().sum(dim=1)
        kept_tokens = kept_mask.sum(dim=1)
        deletion_rate = 1.0 - (kept_tokens / real_tokens.clamp(min=1.0))

        return gate_outputs, kept_mask, deletion_rate

    def apply_hard_deletion(
        self,
        hidden_states: torch.Tensor,
        kept_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Physically remove deleted bytes from the hidden state sequence.

        After hard deletion, different samples in the batch may have
        different lengths. The new sequence length is set by the longest
        remaining sequence, and shorter ones are padded.

        Args:
            hidden_states: Full hidden states before deletion.
                Shape: (batch_size, seq_len, hidden_dim)
            kept_mask: Binary mask from the gate (1=keep, 0=delete).
                Shape: (batch_size, seq_len)

        Returns:
            compressed_states: Hidden states with deleted bytes removed.
                Shape: (batch_size, new_seq_len, hidden_dim)
            new_attention_mask: Updated attention mask for the compressed seq.
                Shape: (batch_size, new_seq_len)
        """
        batch_size = hidden_states.size(0)
        hidden_dim = hidden_states.size(2)

        # Find the maximum number of kept tokens across the batch
        kept_counts = kept_mask.sum(dim=1).long()
        new_seq_len = kept_counts.max().item()

        # Ensure at least 1 token is kept (safety)
        new_seq_len = max(new_seq_len, 1)

        compressed_states = torch.zeros(
            batch_size, new_seq_len, hidden_dim,
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        new_attention_mask = torch.zeros(
            batch_size, new_seq_len,
            device=hidden_states.device, dtype=kept_mask.dtype
        )

        for i in range(batch_size):
            # Gather indices of kept positions
            kept_indices = kept_mask[i].nonzero(as_tuple=True)[0]
            n_kept = kept_indices.size(0)

            if n_kept > 0:
                compressed_states[i, :n_kept] = hidden_states[i, kept_indices]
                new_attention_mask[i, :n_kept] = 1.0

        return compressed_states, new_attention_mask
