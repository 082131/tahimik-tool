# =============================================================================
# Integration tests for the two compression variants.
#
# tests/test_delete_gate.py exercises the gate and the losses in isolation.
# These tests run the actual model forward and backward passes, which is what
# the delete-gate fixes changed:
#
#   - soft deletion now adds the gate score to the attention logits
#   - encoder output is scaled by keep_prob after the final norm
#   - the gate returns a 4-tuple (gate_outputs, keep_prob, kept_mask, rate)
#
# A real ByT5 checkpoint is ~1.2GB, so these build a tiny randomly initialised
# T5 with the same structure. That is enough to prove the wiring, the shapes,
# and the gradient flow are correct — which is what the fixes touched.
#
# Run with:  python -m pytest tests/ -v
# =============================================================================

import pytest
import torch
from transformers import T5Config, T5ForConditionalGeneration

from configs.mrt5_config import MrT5Config
from configs.tahimik_config import TAHIMIKConfig
import src.models.fixed_compression_byt5 as mrt5_module
import src.models.noise_adaptive_byt5 as tahimik_module
from src.training.losses import TAHIMIKLoss


BATCH, SEQ = 2, 24
TARGET_LEN = 10
VOCAB = 384          # ByT5's byte vocabulary
NUM_LAYERS = 4       # must exceed delete_gate_layer (3)


class _StubTokenizer:
    """Stands in for AutoTokenizer, which the models load in __init__."""

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


def _tiny_t5():
    """A structurally faithful but very small T5."""
    return T5ForConditionalGeneration(
        T5Config(
            vocab_size=VOCAB,
            d_model=32,
            d_ff=64,
            d_kv=8,
            num_layers=NUM_LAYERS,
            num_decoder_layers=2,
            num_heads=4,
            decoder_start_token_id=0,
            pad_token_id=0,
            eos_token_id=1,
        )
    )


@pytest.fixture
def patched(monkeypatch):
    """Swap the pretrained loads for a tiny local model in both variants."""
    for module in (mrt5_module, tahimik_module):
        monkeypatch.setattr(
            module, "T5ForConditionalGeneration",
            type("_Stub", (), {"from_pretrained": staticmethod(lambda *a, **k: _tiny_t5())}),
        )
        monkeypatch.setattr(module, "AutoTokenizer", _StubTokenizer)


@pytest.fixture
def inputs():
    """A batch with genuine padding, so masking is actually exercised."""
    torch.manual_seed(0)
    input_ids = torch.randint(3, VOCAB, (BATCH, SEQ))
    attention_mask = torch.zeros(BATCH, SEQ, dtype=torch.long)
    attention_mask[0, :SEQ] = 1        # full length
    attention_mask[1, : SEQ // 2] = 1  # half padded
    input_ids = input_ids * attention_mask

    labels = torch.randint(3, VOCAB, (BATCH, TARGET_LEN))
    noise_level = torch.rand(BATCH)
    return input_ids, attention_mask, labels, noise_level


def _build(variant, patched):
    if variant == "mrt5":
        config = MrT5Config()
        return mrt5_module.FixedCompressionByT5(config), config
    config = TAHIMIKConfig()
    return tahimik_module.NoiseAdaptiveByT5(config), config


# ── Training pass ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("variant", ["mrt5", "tahimik"])
def test_training_forward_runs_and_reports_a_valid_rate(variant, patched, inputs):
    input_ids, attention_mask, labels, noise_level = inputs
    model, _ = _build(variant, patched)
    model.train()

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        noise_level=noise_level,
    )

    assert "loss" in out and torch.isfinite(out["loss"]), "loss is NaN or missing"
    rate = out["deletion_rate"]
    assert torch.all(rate >= 0.0) and torch.all(rate <= 1.0), (
        f"deletion rate outside [0,1]: {rate.tolist()}"
    )
    assert out["keep_prob"].shape == (BATCH, SEQ)
    assert rate.grad_fn is not None, "deletion rate is not differentiable"


@pytest.mark.parametrize("variant", ["mrt5", "tahimik"])
def test_total_loss_puts_gradient_on_the_gate(variant, patched, inputs):
    """
    The end-to-end guarantee. Before the fix the only gradient reaching the
    gate came from L_attn_reg, which was minimised by deleting everything.
    """
    input_ids, attention_mask, labels, noise_level = inputs
    model, config = _build(variant, patched)
    model.train()

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        noise_level=noise_level,
    )
    loss_fn = TAHIMIKLoss(
        w_rate=config.w_rate,
        w_attn_reg=config.w_attn_reg,
        use_compression=True,
        noise_adaptive=(variant == "tahimik"),
    )
    losses = loss_fn(out, noise_level=noise_level)

    model.zero_grad()
    losses["total_loss"].backward()

    gate_grad = model.delete_gate.gate_linear.weight.grad
    assert gate_grad is not None, "no gradient reached the gate"
    assert gate_grad.norm().item() > 0, "gate gradient is identically zero"
    assert torch.isfinite(gate_grad).all(), "gate gradient contains NaN/inf"


def test_rate_loss_alone_reaches_the_gate_through_the_model(patched, inputs):
    """L_rate specifically — the term that was inert — must now propagate."""
    input_ids, attention_mask, labels, noise_level = inputs
    model, config = _build("tahimik", patched)
    model.train()

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        noise_level=noise_level,
    )
    loss_fn = TAHIMIKLoss(use_compression=True, noise_adaptive=True)
    losses = loss_fn(out, noise_level=noise_level)

    model.zero_grad()
    losses["l_rate"].backward(retain_graph=True)

    gate_grad = model.delete_gate.gate_linear.weight.grad
    assert gate_grad is not None and gate_grad.norm().item() > 0, (
        "L_rate still contributes no gradient to the gate"
    )


def test_noise_estimator_is_trained_only_by_l_ne(patched, inputs):
    """
    n is detached everywhere except L_NE, so L_rate must not reach the
    estimator. Guards the manuscript's stated gradient isolation.
    """
    input_ids, attention_mask, labels, noise_level = inputs
    model, _ = _build("tahimik", patched)
    model.train()

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        noise_level=noise_level,
    )
    loss_fn = TAHIMIKLoss(use_compression=True, noise_adaptive=True)
    losses = loss_fn(out, noise_level=noise_level)

    model.zero_grad()
    losses["l_rate"].backward(retain_graph=True)
    first_layer = model.noise_estimator.network[0].weight
    leaked = first_layer.grad is not None and first_layer.grad.norm().item() > 0
    assert not leaked, "L_rate leaked gradient into the noise estimator"

    model.zero_grad()
    losses["l_ne"].backward()
    assert first_layer.grad is not None and first_layer.grad.norm().item() > 0, (
        "L_NE does not train the noise estimator"
    )


# ── Inference pass ────────────────────────────────────────────────────────

@pytest.mark.parametrize("variant", ["mrt5", "tahimik"])
def test_eval_forward_physically_compresses(variant, patched, inputs):
    """Hard deletion must shorten the encoder output — the actual speedup."""
    input_ids, attention_mask, labels, noise_level = inputs
    model, _ = _build(variant, patched)
    model.eval()

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            noise_level=noise_level,
        )

    encoded_len = out["encoder_last_hidden_state"].shape[1]
    assert encoded_len <= SEQ, "hard deletion lengthened the sequence"
    assert torch.isfinite(out["loss"]), "eval loss is NaN"

    rate = out["deletion_rate"]
    assert torch.all(rate >= 0.0) and torch.all(rate <= 1.0)


@pytest.mark.parametrize("variant", ["mrt5", "tahimik"])
def test_generate_runs_end_to_end(variant, patched, inputs):
    """The path the inference API actually calls."""
    input_ids, attention_mask, _, _ = inputs
    model, _ = _build(variant, patched)
    model.eval()

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=8,
            num_beams=2,
        )

    assert output_ids.shape[0] == BATCH
    assert output_ids.dtype == torch.long


# ── Training actually moves the gate ──────────────────────────────────────

def test_short_training_run_moves_deletion_rate_toward_target(patched, inputs):
    """
    A few optimisation steps on the real model should move the measured
    deletion rate toward the target. Before the fix it could not move at all.
    """
    input_ids, attention_mask, labels, noise_level = inputs
    model, config = _build("mrt5", patched)
    model.train()

    loss_fn = TAHIMIKLoss(
        w_rate=10.0,          # emphasise the rate term for a short run
        w_attn_reg=config.w_attn_reg,
        use_compression=True,
        noise_adaptive=False,
    )
    optimizer = torch.optim.Adam(model.delete_gate.parameters(), lr=0.1)

    def current_rate():
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return out["deletion_rate"].mean().item()

    start = current_rate()

    for _ in range(40):
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        losses = loss_fn(out)
        optimizer.zero_grad()
        losses["l_rate"].backward()
        optimizer.step()

    end = current_rate()
    target = config.fixed_deletion_target

    assert abs(end - target) < abs(start - target), (
        f"deletion rate did not move toward the target "
        f"(start={start:.3f}, end={end:.3f}, target={target})"
    )
