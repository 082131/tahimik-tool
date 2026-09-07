import pytest
import torch
import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration

from configs.tahimik_config import TAHIMIKConfig
import src.models.noise_adaptive_byt5 as tahimik_module
from src.models.delete_gate import DeleteGate
from src.training.trainer import Trainer, compute_noise_band_diagnostics


def _tiny_t5():
    return T5ForConditionalGeneration(
        T5Config(
            vocab_size=384,
            d_model=32,
            d_ff=64,
            d_kv=8,
            num_layers=4,
            num_decoder_layers=2,
            num_heads=4,
            decoder_start_token_id=0,
            pad_token_id=0,
            eos_token_id=1,
        )
    )


class _StubTokenizer:
    pad_token_id = 0
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(
        tahimik_module, "T5ForConditionalGeneration",
        type("_Stub", (), {"from_pretrained": staticmethod(lambda *a, **k: _tiny_t5())}),
    )
    monkeypatch.setattr(tahimik_module, "AutoTokenizer", _StubTokenizer)


def test_tahimik_forward_emits_adaptive_diagnostics(patched):
    config = TAHIMIKConfig()
    model = tahimik_module.NoiseAdaptiveByT5(config)
    model.eval()

    batch_size, seq_len = 2, 16
    input_ids = torch.randint(3, 384, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    labels = torch.randint(3, 384, (batch_size, 8))
    noise_level = torch.tensor([0.25, 0.75])

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        noise_level=noise_level,
    )

    assert "adaptive_coefficient" in outputs
    assert outputs["adaptive_coefficient"].item() >= 0.0
    assert "noise_average" in outputs
    assert "noise_scores" in outputs
    assert "deletion_rate" in outputs


def test_noise_band_diagnostics_aggregates_and_keeps_empty_bands_as_none():
    # Noise scores falling into:
    # 0.1  -> band [0.0, 0.2)
    # 0.3  -> band [0.2, 0.4)
    # 0.35 -> band [0.2, 0.4)
    # 0.85 -> band [0.8, 1.0]
    # bands [0.4, 0.6) and [0.6, 0.8) are empty!
    noise_scores = torch.tensor([0.1, 0.3, 0.35, 0.85])
    deletion_rates = torch.tensor([0.4, 0.3, 0.2, 0.1])

    diag = compute_noise_band_diagnostics(noise_scores, deletion_rates)

    assert diag["overall_mean_noise"] == pytest.approx((0.1 + 0.3 + 0.35 + 0.85) / 4)
    assert diag["overall_mean_deletion_rate"] == pytest.approx((0.4 + 0.3 + 0.2 + 0.1) / 4)

    bands = diag["bands"]
    # Band 0: [0.0, 0.2)
    assert bands["[0.0, 0.2)"]["count"] == 1
    assert bands["[0.0, 0.2)"]["mean_deletion"] == pytest.approx(0.4)

    # Band 1: [0.2, 0.4)
    assert bands["[0.2, 0.4)"]["count"] == 2
    assert bands["[0.2, 0.4)"]["mean_deletion"] == pytest.approx((0.3 + 0.2) / 2)

    # Band 2: [0.4, 0.6) -> empty, must be None, NOT 0.0
    assert bands["[0.4, 0.6)"]["count"] == 0
    assert bands["[0.4, 0.6)"]["mean_deletion"] is None

    # Band 3: [0.6, 0.8) -> empty
    assert bands["[0.6, 0.8)"]["count"] == 0
    assert bands["[0.6, 0.8)"]["mean_deletion"] is None

    # Band 4: [0.8, 1.0]
    assert bands["[0.8, 1.0]"]["count"] == 1
    assert bands["[0.8, 1.0]"]["mean_deletion"] == pytest.approx(0.1)


def test_legacy_checkpoint_loading_migrates_cn_and_rejects_negative():
    gate = DeleteGate(hidden_dim=32, noise_adaptive=True)

    # Valid positive legacy cn
    legacy_state = dict(gate.state_dict())
    del legacy_state["raw_cn"]
    legacy_state["cn"] = torch.tensor(1.5)
    gate.load_state_dict(legacy_state)
    assert gate.adaptive_coefficient.item() == pytest.approx(1.5, rel=1e-3)

    # Negative legacy cn must fail closed
    invalid_state = dict(gate.state_dict())
    del invalid_state["raw_cn"]
    invalid_state["cn"] = torch.tensor(-0.5)
    with pytest.raises(ValueError, match="(?i)inverted adaptivity"):
        gate.load_state_dict(invalid_state)


