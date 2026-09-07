# Shared encoder layer runner and position bias propagation.
# Preserves T5 relative position bias across pre-gate and post-gate layers
# and re-indexes position bias after hard byte deletion.

from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class EncoderLayerResult:
    hidden_states: torch.Tensor
    position_bias: torch.Tensor


@dataclass
class HardDeletionResult:
    hidden_states: torch.Tensor
    attention_mask: torch.Tensor
    source_positions: torch.Tensor


def compress_position_bias(
    position_bias: torch.Tensor,
    source_positions: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compress relative position bias to match the pruned sequence length.

    Args:
        position_bias: Tensor of shape (batch, num_heads, seq_len, seq_len).
        source_positions: Tensor of shape (batch, new_seq_len) containing
            original sequence indices for surviving bytes.
        attention_mask: Tensor of shape (batch, new_seq_len) where 1=real, 0=pad.

    Returns:
        Tensor of shape (batch, num_heads, new_seq_len, new_seq_len).
    """
    batch_size, new_seq_len = source_positions.shape
    num_heads = position_bias.shape[1]
    orig_seq_len = position_bias.shape[-1]

    if position_bias.size(0) == 1 and batch_size > 1:
        position_bias = position_bias.expand(batch_size, -1, -1, -1)

    # 1. Gather along query dimension (dim 2)
    query_idx = source_positions.view(batch_size, 1, new_seq_len, 1).expand(
        -1, num_heads, -1, orig_seq_len
    )
    inter = torch.gather(position_bias, 2, query_idx)

    # 2. Gather along key dimension (dim 3)
    key_idx = source_positions.view(batch_size, 1, 1, new_seq_len).expand(
        -1, num_heads, new_seq_len, -1
    )
    compressed = torch.gather(inter, 3, key_idx)

    # 3. Zero rows and columns masked as padding
    valid_2d = (
        attention_mask.unsqueeze(1).unsqueeze(-1)
        * attention_mask.unsqueeze(1).unsqueeze(2)
    ).to(compressed.dtype)
    compressed = compressed * valid_2d

    return compressed


def run_encoder_layers(
    encoder,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    start_layer: int,
    end_layer: int,
    position_bias: Optional[torch.Tensor] = None,
    gate_bias: Optional[torch.Tensor] = None,
) -> EncoderLayerResult:
    """
    Run a subset of encoder layers on hidden states, threading position_bias.

    Args:
        encoder: HuggingFace T5Stack encoder.
        hidden_states: Hidden states tensor of shape (batch, seq_len, hidden_dim).
        attention_mask: Binary mask of shape (batch, seq_len).
        start_layer: Starting layer index (inclusive).
        end_layer: Ending layer index (exclusive).
        position_bias: Optional relative position bias from previous layers.
        gate_bias: Optional additive log-space bias on attention logits (soft deletion).

    Returns:
        EncoderLayerResult containing updated hidden_states and position_bias.
    """
    extended_mask = encoder.get_extended_attention_mask(
        attention_mask, hidden_states.shape[:2]
    )

    if gate_bias is not None:
        extended_mask = extended_mask + gate_bias[:, None, None, :]

    for index in range(start_layer, end_layer):
        layer = encoder.block[index]
        output = layer(
            hidden_states,
            attention_mask=extended_mask,
            position_bias=position_bias,
        )
        hidden_states = output[0]
        if len(output) > 1 and output[1] is not None:
            position_bias = output[1]

    return EncoderLayerResult(hidden_states=hidden_states, position_bias=position_bias)
