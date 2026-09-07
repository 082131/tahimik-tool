import pytest
from unittest.mock import MagicMock, patch
import torch
from fastapi.testclient import TestClient

from backend.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "models" in data
    assert "byt5" in data["models"]
    assert "mrt5" in data["models"]
    assert "tahimik" in data["models"]


def test_normalize_batch_empty_rejected(client):
    response = client.post("/normalize/batch", json={"texts": [], "model": "tahimik"})
    assert response.status_code in (400, 422)


@patch("backend.app.get_model")
def test_normalize_single(mock_get_model, client):
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    mock_tokenizer.batch_decode.return_value = ["normalized text"]

    mock_model = MagicMock()
    mock_model.generate.return_value = torch.tensor([[10, 20]])
    mock_get_model.return_value = (mock_tokenizer, mock_model)

    response = client.post(
        "/normalize",
        json={"text": "raw text", "model": "tahimik"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["input"] == "raw text"
    assert data["normalized"] == "normalized text"
    assert data["model"] == "tahimik"
    assert "inference_time_ms" in data


@patch("backend.app.get_model")
def test_normalize_batch_true_batching(mock_get_model, client):
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2], [3, 4]]),
        "attention_mask": torch.tensor([[1, 1], [1, 1]]),
    }
    mock_tokenizer.batch_decode.return_value = ["norm 1", "norm 2"]

    mock_model = MagicMock()
    mock_model.generate.return_value = torch.tensor([[10, 20], [30, 40]])
    mock_get_model.return_value = (mock_tokenizer, mock_model)

    response = client.post(
        "/normalize/batch",
        json={"texts": ["raw 1", "raw 2"], "model": "tahimik"},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 2
    assert data["results"][0]["input"] == "raw 1"
    assert data["results"][0]["normalized"] == "norm 1"
    assert data["results"][1]["input"] == "raw 2"
    assert data["results"][1]["normalized"] == "norm 2"
    assert "total_time_ms" in data
    # Ensure model.generate was called exactly once for the batch
    assert mock_model.generate.call_count == 1
