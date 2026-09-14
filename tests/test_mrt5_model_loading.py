"""Loading contracts for the Stanford MrT5-backed variants."""

from types import SimpleNamespace
import torch
import torch.nn as nn

from configs.mrt5_config import MrT5Config
from configs.tahimik_config import TAHIMIKConfig
import src.models.fixed_compression as mrt5_module
import src.models.noise_adaptive as tahimik_module
from src.models.delete_gate import DeleteGate, load_mrt5_pretrained_gate


class _TokenizerStub:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


class _RemoteModelStub:
    config = SimpleNamespace(d_model=16)


class _RemoteModelWithEmbeddedGateStub(_RemoteModelStub):
    def __init__(self):
        self.config = SimpleNamespace(d_model=16, delete_gate_layer=2)
        self.encoder = SimpleNamespace(
            block=[SimpleNamespace(has_delete_gate=False) for _ in range(3)]
        )
        self.encoder.block[2].has_delete_gate = True


def test_mrt5_baseline_exposes_the_stanford_custom_model_loader():
    assert hasattr(mrt5_module, "AutoModelForSeq2SeqLM")


def test_mrt5_baseline_loads_stanford_custom_model_code(monkeypatch):
    calls = []
    monkeypatch.setattr(mrt5_module, "AutoTokenizer", _TokenizerStub)
    monkeypatch.setattr(mrt5_module, "load_mrt5_pretrained_gate", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        mrt5_module,
        "AutoModelForSeq2SeqLM",
        SimpleNamespace(
            from_pretrained=lambda model_name, **kwargs: calls.append((model_name, kwargs)) or _RemoteModelStub()
        ),
    )

    mrt5_module.FixedCompressionByT5(MrT5Config())

    assert calls == [("stanfordnlp/mrt5-small", {"trust_remote_code": True})]


def test_tahimik_exposes_the_stanford_custom_model_loader():
    assert hasattr(tahimik_module, "AutoModelForSeq2SeqLM")


def test_tahimik_starts_from_stanford_custom_model_code(monkeypatch):
    calls = []
    monkeypatch.setattr(tahimik_module, "AutoTokenizer", _TokenizerStub)
    monkeypatch.setattr(tahimik_module, "load_mrt5_pretrained_gate", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        tahimik_module,
        "AutoModelForSeq2SeqLM",
        SimpleNamespace(
            from_pretrained=lambda model_name, **kwargs: calls.append((model_name, kwargs)) or _RemoteModelStub()
        ),
    )

    tahimik_module.NoiseAdaptiveByT5(TAHIMIKConfig())

    assert calls == [("stanfordnlp/mrt5-small", {"trust_remote_code": True})]


def test_local_gate_replaces_the_embedded_stanford_gate_without_double_gating(monkeypatch):
    """The wrapper owns one imported gate; the remote stack must not apply a second one."""
    remote_model = _RemoteModelWithEmbeddedGateStub()
    monkeypatch.setattr(mrt5_module, "AutoTokenizer", _TokenizerStub)
    monkeypatch.setattr(mrt5_module, "load_mrt5_pretrained_gate", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        mrt5_module,
        "AutoModelForSeq2SeqLM",
        SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: remote_model),
    )

    mrt5_module.FixedCompressionByT5(MrT5Config())

    assert remote_model.encoder.block[2].has_delete_gate is False


def test_local_gate_copies_weights_from_the_loaded_stanford_gate():
    source_gate = nn.Module()
    source_gate.feed_forward = nn.Linear(16, 1)
    source_gate.layer_norm = nn.LayerNorm(16, elementwise_affine=True)
    with torch.no_grad():
        source_gate.feed_forward.weight.fill_(0.25)
        source_gate.feed_forward.bias.fill_(0.5)

    local_gate = DeleteGate(hidden_dim=16, noise_adaptive=False)

    assert load_mrt5_pretrained_gate(local_gate, "stanfordnlp/mrt5-small", source_gate=source_gate)
    assert torch.allclose(local_gate.gate_linear.weight, source_gate.feed_forward.weight)
    assert torch.allclose(local_gate.gate_linear.bias, source_gate.feed_forward.bias)
