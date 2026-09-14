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


def _compare_model_fixture():
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    tokenizer.batch_decode.return_value = ["normalized text"]
    model = MagicMock()
    model.generate_with_telemetry.return_value = (
        torch.tensor([[10, 20]]),
        [
            {
                "compression_mode": "none",
                "input_token_count": 3,
                "kept_token_count": 3,
                "deleted_token_count": 0,
                "deletion_rate": 0.0,
            }
        ],
    )
    return tokenizer, model


@patch("backend.app.is_available", return_value=True)
@patch("backend.app.get_model")
def test_compare_returns_all_variants_in_study_order_with_utf8_telemetry(
    mock_get_model, _mock_is_available, client
):
    """A comparison is a study-wide result, never a caller-selected subset."""
    mock_get_model.return_value = _compare_model_fixture()

    response = client.post("/compare", json={"text": "raw á"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["input"] == "raw á"
    assert [item["model"] for item in payload["results"]] == [
        "byt5",
        "mrt5",
        "tahimik",
    ]
    assert all(item["normalized"] == "normalized text" for item in payload["results"])
    assert all(item["telemetry"]["input_byte_count"] == 6 for item in payload["results"])


@patch("backend.app.is_available", side_effect=lambda name: name != "mrt5")
@patch("backend.app.get_model")
def test_compare_fails_atomically_when_any_required_checkpoint_is_missing(
    mock_get_model, _mock_is_available, client
):
    """A missing required variant invalidates the comparison rather than a row."""
    response = client.post("/compare", json={"text": "raw text"})

    assert response.status_code == 503
    assert "MrT5" in response.json()["detail"]
    mock_get_model.assert_not_called()


@patch("backend.app.is_available", return_value=True)
@patch("backend.app.get_model")
def test_compare_batch_preserves_input_order_and_returns_every_variant(
    mock_get_model, _mock_is_available, client
):
    """Unlabelled batches remain comparisons, not an implicit evaluation run."""
    mock_get_model.return_value = _compare_model_fixture()

    response = client.post("/compare/batch", json={"texts": ["first", "second"]})

    assert response.status_code == 200
    results = response.json()["results"]
    assert [item["input"] for item in results] == ["first", "second"]
    assert all(
        [model["model"] for model in item["results"]]
        == ["byt5", "mrt5", "tahimik"]
        for item in results
    )
    assert mock_get_model.call_count == 6


@patch("backend.app.is_available", return_value=True)
@patch("fastapi.BackgroundTasks.add_task")
def test_evaluate_queues_only_labelled_test_sets(mock_add_task, _available, client):
    """Evaluation must require input/reference pairs and return a job identifier."""
    response = client.post(
        "/evaluate",
        json={"examples": [{"input": "raw text", "reference": "clean text"}]},
    )

    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    assert response.json()["total_examples"] == 1
    mock_add_task.assert_called_once()


def test_evaluate_rejects_unlabelled_input(client):
    response = client.post("/evaluate", json={"examples": [{"input": "raw text"}]})
    assert response.status_code == 422


def test_evaluate_status_returns_persisted_job_or_404(client, monkeypatch):
    from backend import app as backend_app
    monkeypatch.setattr(backend_app, "EVALUATION_JOBS", {"job-1": {"job_id": "job-1", "status": "running", "stage": "profiling", "total_examples": 2, "completed_examples": 1}})

    response = client.get("/evaluate/job-1")
    assert response.status_code == 200
    assert response.json()["stage"] == "profiling"
    assert client.get("/evaluate/missing").status_code == 404


def test_model_loading_rejects_checkpoint_from_the_wrong_pretrained_source(monkeypatch):
    """Same-size Small checkpoints must not cross the Google/Stanford boundary."""
    from backend import app as backend_app
    from types import SimpleNamespace

    checkpoint = {
        "architecture": {
            "model_name": "google/byt5-small",
            "d_model": 16,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "vocab_size": 32,
        },
        "model_state_dict": {},
    }

    class _Model:
        def __init__(self, _config):
            self.model = SimpleNamespace(config=SimpleNamespace(d_model=16, num_layers=2, num_decoder_layers=2, vocab_size=32))
        def load_state_dict(self, *_args): pass
        def to(self, *_args): return self
        def eval(self): return self
        def parameters(self): return []

    monkeypatch.setattr(backend_app, "_loaded", {})
    monkeypatch.setattr(backend_app, "device", torch.device("cpu"))
    monkeypatch.setattr(backend_app, "checkpoint_path", lambda _name: SimpleNamespace(is_file=lambda: True, __str__=lambda self: "wrong-source.pt"))
    monkeypatch.setattr(backend_app, "load_config", lambda _name: SimpleNamespace(model_name="stanfordnlp/mrt5-small"))
    monkeypatch.setattr(backend_app.importlib, "import_module", lambda _name: SimpleNamespace(FixedCompressionByT5=_Model))
    monkeypatch.setattr(backend_app.torch, "load", lambda *_args, **_kwargs: checkpoint)
    monkeypatch.setitem(__import__("sys").modules, "transformers", SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: object())))

    with pytest.raises(Exception) as exc_info:
        backend_app.get_model("mrt5")
    assert getattr(exc_info.value, "status_code", None) == 503
    assert "incompatible" in str(getattr(exc_info.value, "detail", "")).lower()
