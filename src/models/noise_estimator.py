# =============================================================================
# Learned Noise Estimator — TAHIMIK's Sentence-Level Noise Scorer
#
# The noise estimator is a small feed-forward network that reads the
# contextual hidden states [h] from the early encoder blocks and predicts
# a single noise score n ∈ [0, 1] for the entire sentence.
#
# Architecture (from the manuscript, Noise-Adaptive Deletion section):
#   1. Mean-pool all non-padding hidden states into one sentence vector
#   2. Pass through a two-layer feed-forward network
#   3. Apply sigmoid to produce the noise score
#
# The noise score n controls the delete gate's compression behavior:
#   - n ≈ 0 (clean input)  → aggressive compression allowed
#   - n ≈ 1 (noisy input)  → compression is conservative
#
# Training signal: L_NE compares predicted n against n* (the byte-level
# edit distance ratio computed during data preparation). This is the ONLY
# loss term that trains the noise estimator — n is detached everywhere
# else so the gate's gradients do not leak into the estimator.
# =============================================================================

import torch
import torch.nn as nn


class NoiseEstimator(nn.Module):
    """
    Predicts a sentence-level noise score from encoder hidden states.

    This module is the first sub-component of the Noise-Adaptive Deletion
    Module. Its output conditions the delete gate's behavior.

    Args:
        hidden_dim: Dimensionality of the encoder hidden states (d_model).
        intermediate_dim: Hidden layer size of the feed-forward network.
    """

    def __init__(self, hidden_dim: int, intermediate_dim: int = 256):
        super().__init__()

        # Two-layer MLP: hidden_dim → intermediate_dim → 1
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(intermediate_dim, 1),
        )

        # Final sigmoid maps the output to [0, 1]
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the noise score for each sentence in the batch.

        Args:
            hidden_states: Encoder hidden states after the early blocks.
                Shape: (batch_size, seq_len, hidden_dim)
            attention_mask: Binary mask where 1 = real token, 0 = padding.
                Shape: (batch_size, seq_len)

        Returns:
            noise_scores: Per-sentence noise scores in [0, 1].
                Shape: (batch_size,)
        """
        # Expand mask for broadcasting with hidden_dim
        # Shape: (batch_size, seq_len, 1)
        mask_expanded = attention_mask.unsqueeze(-1).float()

        # Mean-pool over non-padding positions
        # Numerator: sum of hidden states at real positions
        summed = (hidden_states * mask_expanded).sum(dim=1)

        # Denominator: number of real tokens per sentence
        # Clamp to avoid division by zero for fully-padded sequences
        counts = mask_expanded.sum(dim=1).clamp(min=1.0)

        # Sentence vector: (batch_size, hidden_dim)
        sentence_vector = summed / counts

        # MLP + sigmoid → noise score: (batch_size, 1) → (batch_size,)
        noise_scores = self.sigmoid(self.network(sentence_vector)).squeeze(-1)

        return noise_scores
