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


def gumbel_noise_like(x: torch.Tensor) -> torch.Tensor:
    """
    Sample Gumbel(0, 1) noise shaped like x.

    Adapted from the MrT5 reference implementation (jkallini/mrt5,
    Apache 2.0) — see ATTRIBUTIONS.md. The epsilon guards the log from
    underflowing to -inf, and is loosened under fp16 where the smallest
    representable positive number is much larger.
    """
    eps = 3e-4 if x.dtype == torch.float16 else 1e-10
    uniform = torch.empty_like(x).uniform_(eps, 1 - eps)
    return -(-uniform.log()).log()


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
        use_gumbel_noise: bool = True,
    ):
        super().__init__()

        self.k = k
        self.noise_adaptive = noise_adaptive
        self.noise_avg_momentum = noise_avg_momentum
        self.use_gumbel_noise = use_gumbel_noise

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute gate scores and the resulting keep probabilities.

        Args:
            hidden_states: Encoder hidden states at the gate layer.
                Shape: (batch_size, seq_len, hidden_dim)
            attention_mask: Binary mask (1=real, 0=pad).
                Shape: (batch_size, seq_len)
            noise_scores: Per-sentence noise scores from the noise estimator.
                Shape: (batch_size,). Required if noise_adaptive=True.

        Returns:
            gate_outputs: Per-byte gate scores in [k, 0]. Used as an additive
                log-space bias on attention logits during training.
                Shape: (batch_size, seq_len, 1)
            keep_prob: Differentiable keep probability in [0, 1], derived from
                the gate score. Shape: (batch_size, seq_len)
            kept_mask: Binary mask of bytes that survive HARD deletion.
                Shape: (batch_size, seq_len)
            deletion_rate: Fraction of bytes deleted per sentence, in [0, 1].
                Differentiable during training (computed from keep_prob) so
                that L_rate can actually train the gate; computed from the
                hard mask during inference so the reported rate is the real
                one. Shape: (batch_size,)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # ── Step 1: Compute raw gate scores ─────────────────────────────
        # LayerNorm → Linear → rescaled sigmoid in [k, 0]
        normed = self.layer_norm(hidden_states)
        logits = self.gate_linear(normed)  # (batch, seq, 1)

        # Gumbel noise on the logits during training, from the MrT5 reference
        # implementation (see ATTRIBUTIONS.md). Keep/delete is close to a
        # discrete decision, and without perturbation the gate can settle on a
        # choice early and never sample the alternative hard enough to learn
        # whether it was better. Training only — inference must be
        # deterministic, or the same sentence would compress differently on
        # each call.
        if self.training and self.use_gumbel_noise:
            logits = logits + gumbel_noise_like(logits)

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

        # ── Step 3: Keep probability (differentiable) ───────────────────
        # Map the gate score from [k, 0] onto a keep probability in [0, 1]:
        #   gate = 0  → p = 1  (definitely keep)
        #   gate = k  → p = 0  (definitely delete)
        # This is the differentiable quantity the rate loss is computed from.
        keep_prob = 1.0 + (gate_outputs.squeeze(-1) / abs(self.k))
        keep_prob = keep_prob * attention_mask.float()  # padding never counts

        # ── Step 4: Hard keep/delete decision ───────────────────────────
        # Used for physical deletion at inference and for reporting the true
        # rate. Non-differentiable by construction (it is a threshold).
        kept_mask = (gate_outputs.squeeze(-1) > self.hard_threshold).float()
        kept_mask = kept_mask * attention_mask.float()

        # ── Step 5: Deletion rate ───────────────────────────────────────
        # real_tokens excludes padding, so the rate is always a genuine
        # fraction of the sentence in [0, 1].
        real_tokens = attention_mask.float().sum(dim=1).clamp(min=1.0)

        if self.training:
            # Soft (differentiable) rate — this is what makes L_rate able to
            # push the gate toward its target. Using the hard mask here would
            # produce a constant with no gradient.
            deletion_rate = 1.0 - (keep_prob.sum(dim=1) / real_tokens)
        else:
            # Hard rate — the compression actually achieved at inference.
            deletion_rate = 1.0 - (kept_mask.sum(dim=1) / real_tokens)

        return gate_outputs, keep_prob, kept_mask, deletion_rate

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
        batch_size, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device

        keep = kept_mask.bool()
        kept_counts = keep.sum(dim=1)

        # At least one position, so the decoder always receives something even
        # if the gate deleted an entire sentence.
        new_seq_len = max(int(kept_counts.max().item()), 1)

        # Vectorised gather, adapted from the MrT5 reference implementation
        # (see ATTRIBUTIONS.md). The previous version looped over the batch in
        # Python. That runs at inference, which is exactly what the efficiency
        # research question measures, so the loop's overhead landed on the two
        # compressed variants and understated the very saving they exist to
        # demonstrate.
        #
        # cumsum over the keep mask gives each kept position its destination
        # index; deleted positions repeat the previous index and contribute a
        # source index of 0, which scatter_add_ then adds harmlessly.
        target_pos = (torch.cumsum(keep.long(), dim=1) - 1).clamp(min=0)

        positions = torch.arange(seq_len, device=device).expand(batch_size, -1)
        positions = positions * keep.long()

        src_positions = torch.zeros(
            batch_size, new_seq_len, device=device, dtype=torch.long
        )
        src_positions.scatter_add_(1, target_pos, positions)

        compressed_states = torch.gather(
            hidden_states, 1,
            src_positions.unsqueeze(-1).expand(-1, -1, hidden_dim),
        )

        # Positions beyond a row's kept count are padding.
        new_attention_mask = (
            torch.arange(new_seq_len, device=device).expand(batch_size, -1)
            < kept_counts.unsqueeze(1)
        ).to(kept_mask.dtype)

        # Zero the padded tail so it carries no stale hidden state.
        compressed_states = compressed_states * new_attention_mask.unsqueeze(-1).to(
            compressed_states.dtype
        )

        return compressed_states, new_attention_mask
