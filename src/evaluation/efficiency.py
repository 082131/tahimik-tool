import time
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from src.data.dataset import NormalizationDataset, NormalizationCollator, collate_fn


class EfficiencyBenchmark:
    """Benchmark runner for measuring inference latency and GPU memory usage.

    Designed for controlled, reproducible evaluation:
      - Enforces batch_size=1 to capture realistic interactive per-sentence latency.
      - Executes warmup passes first so cold-start JIT and CUDA kernel overhead
        don't skew measurements.
      - Uses torch.cuda.synchronize() around model.generate() so we measure actual
        hardware execution rather than asynchronous kernel launch time.
      - Tracks per-sentence latency as well as run-level peak GPU memory (in MB).
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device,
        num_beams: int = 4,
        warmup_passes: int = 5,
        inference_runs: int = 20,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.num_beams = num_beams
        self.warmup_passes = warmup_passes
        self.inference_runs = inference_runs

    @torch.no_grad()
    def _run_inference(self, dataloader: DataLoader) -> List[float]:
        """Run a single pass through the dataloader and measure per-item latency."""
        self.model.eval()
        durations = []

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            # CUDA operations are asynchronous by default. If we don't synchronize
            # here, time.perf_counter() will only measure how fast PyTorch dispatched
            # the kernel to the GPU queue, not how long it took to run.
            if self.device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=1024,
                num_beams=self.num_beams,
            )

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start_time
            durations.append(elapsed)

        return durations

    def benchmark(
        self,
        test_dataset: NormalizationDataset,
        batch_size: int = 1,
    ) -> Dict[str, Any]:
        """Run the full efficiency benchmark on the given dataset.

        Args:
            test_dataset: Dataset containing test inputs.
            batch_size: Batch size for inference. Must be 1 to match Chapter 3
                per-sentence latency methodology.

        Returns:
            Dict containing average time, standard deviation, per-sentence timings,
            and peak GPU memory usage across runs.
        """
        if batch_size != 1:
            raise ValueError(
                "Chapter 3 per-sentence latency evaluation strictly requires batch_size=1"
            )

        pad_id = getattr(self.tokenizer, "pad_token_id", 0)
        loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=NormalizationCollator(pad_token_id=pad_id),
        )

        num_sentences = len(test_dataset)

        # 1. Warm-up passes: run through the dataset a few times to ensure CUDA
        # contexts, kernels, and memory pools are initialized before timing.
        for _ in range(self.warmup_passes):
            self._run_inference(loader)

        # 2. Timed inference runs: collect latency and memory per run
        timed_runs: List[List[float]] = []
        memory_runs: List[float] = []

        for _ in range(self.inference_runs):
            # Reset peak memory statistics before each run so we record the peak
            # strictly for this specific pass.
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)

            durations = self._run_inference(loader)
            timed_runs.append(durations)

            if self.device.type == "cuda":
                peak_bytes = torch.cuda.max_memory_allocated(self.device)
                memory_runs.append(peak_bytes / (1024 * 1024))

        # 3. Compute per-sentence latency averaged across all timed runs
        if num_sentences > 0:
            per_sentence_time = [
                sum(run[i] for run in timed_runs) / len(timed_runs)
                for i in range(num_sentences)
            ]
        else:
            per_sentence_time = []

        # Mean latency for each completed run
        run_means = [
            sum(run) / max(len(run), 1)
            for run in timed_runs
        ]

        # Overall summary statistics
        avg_time = float(sum(run_means) / max(len(run_means), 1))
        std_time = (
            float(torch.tensor(run_means).std().item())
            if len(run_means) > 1
            else 0.0
        )

        mean_peak_gpu_memory = (
            float(sum(memory_runs) / len(memory_runs)) if memory_runs else None
        )
        std_peak_gpu_memory = (
            float(torch.tensor(memory_runs).std().item()) if len(memory_runs) > 1 else 0.0
        ) if memory_runs else None
        max_peak_gpu_memory = max(memory_runs) if memory_runs else None

        return {
            "avg_time_per_sentence": avg_time,
            "std_time_per_sentence": std_time,
            "per_sentence_time_seconds": per_sentence_time,
            "run_mean_time_seconds": run_means,
            # `peak_gpu_memory_mb` remains for consumers that expect one value;
            # it is the mean of the per-run peaks required by Chapter 3.
            "peak_gpu_memory_mb": mean_peak_gpu_memory,
            "peak_gpu_memory_runs_mb": memory_runs,
            "mean_peak_gpu_memory_mb": mean_peak_gpu_memory,
            "std_peak_gpu_memory_mb": std_peak_gpu_memory,
            "max_peak_gpu_memory_mb": max_peak_gpu_memory,
            "gpu_memory_available": bool(memory_runs),
            "num_sentences": num_sentences,
        }
