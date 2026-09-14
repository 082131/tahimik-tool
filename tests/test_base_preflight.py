from unittest.mock import MagicMock, patch
import pytest
import torch

from scripts.preflight_base import run_preflight


def test_preflight_metadata_mode_runs_offline():
    result = run_preflight(device="cpu", max_input_length=1024, metadata_only=True)
    assert result["success"] is True
    assert result["all_models_valid"] is True
    assert "variants" in result
    expected_models = {
        "byt5": "google/byt5-small",
        "mrt5": "stanfordnlp/mrt5-small",
        "tahimik": "stanfordnlp/mrt5-small",
    }
    for v, model_name in expected_models.items():
        assert v in result["variants"]
        assert result["variants"][v]["model_name"] == model_name
        assert result["variants"][v]["is_expected_backbone"] is True
        assert result["variants"][v]["stage1_effective_batch"] == 16
        assert result["variants"][v]["stage2_effective_batch"] == 8


def test_preflight_rejects_non_small_backbone(monkeypatch):
    import scripts.preflight_base as pb
    mock_cfg = MagicMock()
    mock_cfg.model_name = "google/byt5-large"
    mock_cfg.stage1_batch_size = 4
    mock_cfg.stage1_gradient_accumulation_steps = 4
    mock_cfg.stage2_batch_size = 4
    mock_cfg.stage2_gradient_accumulation_steps = 2
    monkeypatch.setitem(pb.VARIANTS, "byt5", (lambda: mock_cfg, MagicMock))

    result = run_preflight(device="cpu", max_input_length=1024, metadata_only=True)
    assert result["success"] is False
    assert result["all_models_valid"] is False


@patch("scripts.preflight_base.load_variant_model")
def test_preflight_oom_handling_recommends_batch_accumulation(mock_load):
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    mock_model = MagicMock()
    mock_model.parameters.return_value = [torch.tensor([1.0])]
    mock_model.generate.side_effect = torch.cuda.OutOfMemoryError("CUDA out of memory")
    mock_load.return_value = (mock_tokenizer, mock_model)

    result = run_preflight(device="cpu", max_input_length=1024, metadata_only=False)
    assert result["success"] is False
    assert result.get("oom_detected") is True
    assert "gradient accumulation" in result.get("recommendation", "").lower()
