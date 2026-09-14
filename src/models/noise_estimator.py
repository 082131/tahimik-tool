# 2-layer MLP that pools encoder hidden states and predicts sentence noise level (0.0 to 1.0).

import torch
import torch.nn as nn


class NoiseEstimator(nn.Module):
    """
    Predicts sentence-level noise scores from early encoder hidden states.

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
