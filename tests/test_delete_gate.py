# =============================================================================
# Regression tests for the delete gate's training signal.
#
# These lock in three bugs that made the compression mechanism silently
# inert. Each test fails against the pre-fix implementation.
#
#   1. L_rate carried no gradient, so the gate never learned a deletion rate.
#      `kept_mask` was a constant tensor of ones during training, which also
#      counted padding as "kept" and produced NEGATIVE deletion rates.
#
#   2. L_attn_reg was minimised by deleting the entire sequence, and it was
#      the only gradient reaching the gate.
#
#   3. The "soft" mask was multiplied into the binary attention mask, which
#      HuggingFace converts via `(1 - mask) * finfo.min`. A keep probability
#      of 0.99997 became a bias of -1.1e34 — i.e. a hard delete.
#
# Run with:  python -m pytest tests/ -v
# =============================================================================

import torch
import pytest

from src.models.delete_gate import DeleteGate
from src.training.losses import TAHIMIKLoss


BATCH, SEQ, DIM = 4, 32, 16
LENGTHS = [20, 25, 12, 32]   # real (unpadded) length of each sentence


@pytest.fixture
def batch():
    """A batch with realistic, varied padding."""
    torch.manual_seed(0)
    hidden = torch.randn(BATCH, SEQ, DIM)
    mask = torch.zeros(BATCH, SEQ)
    for i, length in enumerate(LENGTHS):
        mask[i, :length] = 1.0
    noise = torch.rand(BATCH)
    return hidden, mask, noise


@pytest.fixture
def gate():
    return DeleteGate(hidden_dim=DIM, k=-30.0, noise_adaptive=True)


# ── Bug 1: deletion rate is a real fraction, and it carries gradient ──────

def test_deletion_rate_is_a_valid_fraction_in_training(gate, batch):
    """Padding must not be counted as kept — that produced negative rates."""
    hidden, mask, noise = batch
    gate.train()
    _, _, _, rate = gate(hidden, mask, noise_scores=noise)

    assert torch.all(rate >= 0.0), f"negative deletion rate: {rate.tolist()}"
    assert torch.all(rate <= 1.0), f"deletion rate above 1: {rate.tolist()}"


def test_deletion_rate_is_a_valid_fraction_in_eval(gate, batch):
    hidden, mask, noise = batch
    gate.eval()
    _, _, _, rate = gate(hidden, mask, noise_scores=noise)

    assert torch.all(rate >= 0.0)
    assert torch.all(rate <= 1.0)


def test_deletion_rate_is_differentiable_in_training(gate, batch):
    """Without this, L_rate is a constant and trains nothing."""
    hidden, mask, noise = batch
    gate.train()
    _, _, _, rate = gate(hidden, mask, noise_scores=noise)

    assert rate.grad_fn is not None, "deletion_rate has no gradient path"


def test_rate_loss_produces_gradient_on_the_gate(gate, batch):
    """The end-to-end guarantee: L_rate can actually move the gate weights."""
    hidden, mask, noise = batch
    gate.train()
    gate_out, keep_prob, _, rate = gate(hidden, mask, noise_scores=noise)

    loss_fn = TAHIMIKLoss(use_compression=True, noise_adaptive=True)
    losses = loss_fn(
        {
            "loss": torch.tensor(0.0, requires_grad=True),
            "deletion_rate": rate,
            "target_deletion_rate": 0.5 * (1.0 - noise),
            "gate_outputs": gate_out,
            "keep_prob": keep_prob,
            "noise_scores": noise,
        },
        noise_level=noise,
    )

    gate.zero_grad()
    losses["l_rate"].backward()

    grad = gate.gate_linear.weight.grad
    assert grad is not None, "L_rate produced no gradient on the gate"
    assert grad.norm().item() > 0, "L_rate gradient is identically zero"


def test_rate_loss_drives_deletion_rate_toward_its_target(gate, batch):
    """Optimising L_rate alone should converge the rate onto the target."""
    hidden, mask, noise = batch
    target = torch.full((BATCH,), 0.5)
    optimizer = torch.optim.Adam(gate.parameters(), lr=0.05)
    loss_fn = TAHIMIKLoss(use_compression=True, noise_adaptive=False)

    gate.train()
    for _ in range(300):
        gate_out, keep_prob, _, rate = gate(hidden, mask, noise_scores=noise)
        losses = loss_fn(
            {
                "loss": torch.tensor(0.0, requires_grad=True),
                "deletion_rate": rate,
                "gate_outputs": gate_out,
                "keep_prob": keep_prob,
                "fixed_deletion_target": 0.5,
            }
        )
        optimizer.zero_grad()
        losses["l_rate"].backward()
        optimizer.step()

    _, _, _, final_rate = gate(hidden, mask, noise_scores=noise)
    assert torch.allclose(final_rate, target, atol=0.05), (
        f"rate did not converge to 0.5: {final_rate.tolist()}"
    )


# ── Bug 2: the regularizer must not favour deleting everything ────────────

def test_attn_reg_is_symmetric_between_keep_and_delete(batch):
    """
    The old term was minimised at p=0 (delete all). The replacement must
    treat p=0 and p=1 identically and penalise only indecision at p=0.5.
    """
    loss_fn = TAHIMIKLoss(use_compression=True, noise_adaptive=False)

    def attn_reg_for(p_value):
        keep_prob = torch.full((BATCH, SEQ), p_value)
        return loss_fn(
            {
                "loss": torch.tensor(0.0, requires_grad=True),
                "deletion_rate": torch.zeros(BATCH),
                "gate_outputs": torch.zeros(BATCH, SEQ, 1),
                "keep_prob": keep_prob,
                "fixed_deletion_target": 0.0,
            }
        )["l_attn_reg"].item()

    delete_all, keep_all, undecided = attn_reg_for(0.0), attn_reg_for(1.0), attn_reg_for(0.5)

    assert delete_all == pytest.approx(keep_all, abs=1e-6), (
        "regularizer prefers one decision over the other"
    )
    assert undecided > delete_all, "indecision (p=0.5) must be penalised most"


def test_attn_reg_alone_does_not_collapse_the_gate(gate, batch):
    """
    Previously, optimising L_attn_reg alone drove deletion to 100%.
    It may now sharpen the gate, but must not wipe out the sequence.
    """
    hidden, mask, noise = batch
    optimizer = torch.optim.SGD(gate.parameters(), lr=0.5)
    loss_fn = TAHIMIKLoss(use_compression=True, noise_adaptive=False)

    gate.train()
    for _ in range(200):
        gate_out, keep_prob, _, rate = gate(hidden, mask, noise_scores=noise)
        losses = loss_fn(
            {
                "loss": torch.tensor(0.0, requires_grad=True),
                "deletion_rate": rate,
                "gate_outputs": gate_out,
                "keep_prob": keep_prob,
                "fixed_deletion_target": 0.5,
            }
        )
        optimizer.zero_grad()
        losses["l_attn_reg"].backward()
        optimizer.step()

    gate.eval()
    _, _, _, final_rate = gate(hidden, mask, noise_scores=noise)
    assert final_rate.mean().item() < 0.99, (
        f"gate collapsed to deleting everything: {final_rate.mean().item():.3f}"
    )


# ── Bug 3: keep_prob must be a usable soft signal ─────────────────────────

def test_keep_prob_is_bounded_and_differentiable(gate, batch):
    hidden, mask, noise = batch
    gate.train()
    _, keep_prob, _, _ = gate(hidden, mask, noise_scores=noise)

    assert keep_prob.shape == (BATCH, SEQ)
    assert torch.all(keep_prob >= 0.0) and torch.all(keep_prob <= 1.0)
    assert keep_prob.grad_fn is not None


def test_keep_prob_is_zero_on_padding(gate, batch):
    """Padding must never contribute to the rate calculation."""
    hidden, mask, noise = batch
    gate.train()
    _, keep_prob, _, _ = gate(hidden, mask, noise_scores=noise)

    for i, length in enumerate(LENGTHS):
        assert torch.all(keep_prob[i, length:] == 0.0), (
            f"row {i} has nonzero keep_prob on padding"
        )


def test_gate_bias_is_additive_not_multiplicative():
    """
    Documents why the gate is added to attention logits. Multiplying a near-1
    keep probability into the binary mask and letting HuggingFace convert it
    yields a bias of ~-1e34, which softmax reads as -inf.
    """
    finfo_min = torch.finfo(torch.float32).min
    keep_prob = 0.99997          # gate = -0.001: "definitely keep"

    broken_bias = (1.0 - keep_prob) * finfo_min
    broken_weight = torch.softmax(torch.tensor([broken_bias, 0.0]), dim=0)[0]
    assert broken_weight.item() == pytest.approx(0.0, abs=1e-9)

    correct_bias = -0.001        # the gate value itself, added to the logit
    correct_weight = torch.softmax(torch.tensor([correct_bias, 0.0]), dim=0)[0]
    assert correct_weight.item() > 0.49


# ── Noise-adaptive behaviour (the thesis contribution) ────────────────────

def test_noisier_sentences_are_compressed_less(gate, batch):
    """
    The core claim: a higher noise score shifts gate scores up, so fewer
    bytes are deleted. Compares the same batch under clean vs noisy scores.
    """
    hidden, mask, _ = batch
    gate.eval()

    clean = torch.full((BATCH,), 0.05)
    noisy = torch.full((BATCH,), 0.95)

    _, _, _, rate_clean = gate(hidden, mask, noise_scores=clean)
    _, _, _, rate_noisy = gate(hidden, mask, noise_scores=noisy)

    assert rate_noisy.mean().item() <= rate_clean.mean().item(), (
        f"noisy input was compressed MORE ({rate_noisy.mean():.3f}) than "
        f"clean ({rate_clean.mean():.3f}) — the adaptive shift is inverted"
    )


# ── Techniques adopted from the MrT5 reference implementation ─────────────
#
# Both of these come from jkallini/mrt5 (Apache 2.0, see ATTRIBUTIONS.md).
# They were added after the fact rather than test-first, which inverts
# Constitution Principle IV — recorded rather than hidden.


def test_gumbel_noise_perturbs_gate_scores_in_training_only(batch):
    """
    The reference implementation adds Gumbel noise to the gate logits during
    training so the keep/delete decision stays explorable. It must NOT apply
    at inference: the same sentence has to compress identically on every
    call, or reported efficiency depends on luck.
    """
    hidden, mask, _ = batch
    gate = DeleteGate(hidden_dim=DIM, noise_adaptive=False, use_gumbel_noise=True)

    gate.train()
    a = gate(hidden, mask)[0]
    b = gate(hidden, mask)[0]
    assert not torch.allclose(a, b), (
        "gate scores identical across two training passes — Gumbel noise is not applied"
    )

    gate.eval()
    c = gate(hidden, mask)[0]
    d = gate(hidden, mask)[0]
    assert torch.allclose(c, d), (
        "gate scores differ between two eval passes — inference is not deterministic"
    )


def test_gumbel_noise_can_be_disabled(batch):
    """The flag is a config value, not a hardcoded behaviour."""
    hidden, mask, _ = batch
    gate = DeleteGate(hidden_dim=DIM, noise_adaptive=False, use_gumbel_noise=False)
    gate.train()

    a = gate(hidden, mask)[0]
    b = gate(hidden, mask)[0]
    assert torch.allclose(a, b), "noise applied despite use_gumbel_noise=False"


def test_hard_deletion_keeps_exactly_the_marked_positions(batch):
    """
    Hard deletion was rewritten from a Python loop to a vectorised gather,
    because it runs at inference — which is precisely what the efficiency
    research question measures. Behaviour must be unchanged: the surviving
    hidden states are the kept ones, in order, left-packed.
    """
    hidden, mask, _ = batch
    torch.manual_seed(7)
    kept = ((torch.rand(BATCH, SEQ) > 0.5).float() * mask)

    gate = DeleteGate(hidden_dim=DIM, noise_adaptive=False)
    compressed, new_mask = gate.apply_hard_deletion(hidden, kept)

    for i in range(BATCH):
        idx = kept[i].nonzero(as_tuple=True)[0]
        n = idx.size(0)
        assert new_mask[i, :n].sum() == n, f"row {i}: mask lost a kept position"
        assert new_mask[i, n:].sum() == 0, f"row {i}: mask marks padding as real"
        assert torch.allclose(compressed[i, :n], hidden[i, idx]), (
            f"row {i}: compressed states are not the kept states in order"
        )


def test_hard_deletion_survives_a_fully_deleted_sentence(batch):
    """
    If the gate deletes an entire sentence the decoder must still receive
    something, or generation has nothing to attend to.
    """
    hidden, _, _ = batch
    kept = torch.zeros(BATCH, SEQ)
    kept[1, :5] = 1.0          # one row keeps something, the rest keep nothing

    gate = DeleteGate(hidden_dim=DIM, noise_adaptive=False)
    compressed, new_mask = gate.apply_hard_deletion(hidden, kept)

    assert compressed.size(1) >= 1, "compressed sequence length collapsed to zero"
    assert new_mask[0].sum() == 0, "an all-deleted row reports surviving positions"
    assert new_mask[1].sum() == 5, "the row that kept 5 positions did not keep 5"
