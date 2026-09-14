import pytest
import torch
import torch.nn as nn
from src.evaluation.efficiency import EfficiencyBenchmark
from src.data.dataset import NormalizationDataset


class DummyTokenizer:
    pad_token_id = 0

    def __call__(self, text, max_length=1024, padding=False, truncation=True, return_tensors="pt"):
        raw_bytes = list(text.encode("utf-8"))
        if truncation and len(raw_bytes) > max_length:
            raw_bytes = raw_bytes[:max_length]
        return {
            "input_ids": torch.tensor([raw_bytes], dtype=torch.long),
            "attention_mask": torch.tensor([[1] * len(raw_bytes)], dtype=torch.long),
        }


class DummyModel(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def generate(self, input_ids, attention_mask=None, max_length=1024, num_beams=1):
        # Return simple generated sequence of same batch size
        return torch.tensor([[10, 11, 12]], dtype=torch.long)


@pytest.fixture
def benchmark_setup():
    tokenizer = DummyTokenizer()
    model = DummyModel(tokenizer)
    device = torch.device("cpu")
    benchmark = EfficiencyBenchmark(
        model=model,
        tokenizer=tokenizer,
        device=device,
        num_beams=1,
        warmup_passes=1,
        inference_runs=2,
    )
    dataset = NormalizationDataset(
        ["short input", "much longer input for test"],
        ["short input", "much longer input for test"],
        tokenizer,
    )
    return benchmark, dataset


def test_efficiency_enforces_batch_size_one(benchmark_setup):
    benchmark, dataset = benchmark_setup
    with pytest.raises(ValueError, match="strictly requires batch_size=1"):
        benchmark.benchmark(dataset, batch_size=2)


def test_efficiency_runs_with_variable_length_dataset(benchmark_setup):
    benchmark, dataset = benchmark_setup
    results = benchmark.benchmark(dataset, batch_size=1)

    assert "avg_time_per_sentence" in results
    assert "std_time_per_sentence" in results
    assert "per_sentence_time_seconds" in results
    assert len(results["per_sentence_time_seconds"]) == len(dataset)
    assert results["avg_time_per_sentence"] >= 0.0


def test_cpu_benchmark_marks_all_peak_memory_summaries_unavailable(benchmark_setup):
    benchmark, dataset = benchmark_setup

    results = benchmark.benchmark(dataset, batch_size=1)

    assert results["gpu_memory_available"] is False
    assert results["mean_peak_gpu_memory_mb"] is None
    assert results["std_peak_gpu_memory_mb"] is None
    assert results["max_peak_gpu_memory_mb"] is None
