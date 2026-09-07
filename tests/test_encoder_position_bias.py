import torch
import pytest
from transformers.models.t5.modeling_t5 import T5Config, T5Block, T5Stack

from src.models.encoder_layers import (
    run_encoder_layers,
    compress_position_bias,
    EncoderLayerResult,
)


class InstrumentedTestBlock(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.received_position_bias = None

    def forward(self, hidden_states, attention_mask=None, position_bias=None, **kwargs):
        self.received_position_bias = position_bias
        return self.block(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            **kwargs,
        )


class MockTinyEncoder(torch.nn.Module):
    def __init__(self, num_layers=4, d_model=32, num_heads=4):
        super().__init__()
        config = T5Config(d_model=d_model, d_kv=8, num_heads=num_heads)
        self.block = torch.nn.ModuleList([
            InstrumentedTestBlock(T5Block(config, has_relative_attention_bias=(i == 0)))
            for i in range(num_layers)
        ])

    def get_extended_attention_mask(self, attention_mask, input_shape):
        batch_size, seq_length = input_shape
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask


@pytest.fixture
def tiny_encoder():
    return MockTinyEncoder(num_layers=4, d_model=32, num_heads=4)


def test_shared_runner_threads_position_bias_between_layers(tiny_encoder):
    batch_size, seq_len, d_model = 2, 8, 32
    hidden_states = torch.randn(batch_size, seq_len, d_model)
    mask = torch.ones(batch_size, seq_len)

    result = run_encoder_layers(
        tiny_encoder,
        hidden_states,
        mask,
        start_layer=0,
        end_layer=4,
    )

    assert isinstance(result, EncoderLayerResult)
    assert result.position_bias is not None
    assert result.position_bias.shape[-2:] == (seq_len, seq_len)

    # Layer 0 had relative attention bias, so it computes bias (received None)
    assert tiny_encoder.block[0].received_position_bias is None
    # Layers 1, 2, 3 must receive the computed position_bias from layer 0
    for i in range(1, 4):
        received = tiny_encoder.block[i].received_position_bias
        assert received is not None, f"Layer {i} received None position_bias!"
        assert received.shape[-2:] == (seq_len, seq_len)


def test_compressed_position_bias_matches_retained_positions():
    bias = torch.arange(2 * 4 * 4).reshape(1, 2, 4, 4).float()
    source_positions = torch.tensor([[0, 2]])
    mask = torch.tensor([[1, 1]])
    compressed = compress_position_bias(bias, source_positions, mask)
    expected = bias[:, :, [0, 2]][:, :, :, [0, 2]]
    assert torch.equal(compressed, expected)


def test_compressed_position_bias_zeroes_padding():
    bias = torch.ones(1, 2, 4, 4)
    source_positions = torch.tensor([[0, 1, 2]])
    # Last position is padding (0 in mask)
    mask = torch.tensor([[1, 1, 0]])
    compressed = compress_position_bias(bias, source_positions, mask)
    # Row 2 and Column 2 should be zeroed
    assert torch.all(compressed[:, :, 2, :] == 0.0)
    assert torch.all(compressed[:, :, :, 2] == 0.0)
    assert torch.all(compressed[:, :, :2, :2] == 1.0)
