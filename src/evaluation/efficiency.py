# =============================================================================
# Efficiency Benchmarking for TAHIMIK
#
# Two efficiency metrics from the manuscript (Evaluation Criteria for
# Computational Efficiency section):
#
#   Average Inference Time — Mean wall-clock time per sentence during
#       inference. Measured over 20 runs after 5 CUDA warmup passes,
#       averaged across the test set.
#
#   Peak GPU Memory — Maximum GPU memory allocated during a forward +
#       generate pass. Measured using torch.cuda.max_memory_allocated()
#       across 20 runs.
#
# The same hardware (Colab A100) and batch size are used for all three
# model variants (control variable).
# =============================================================================

import time
import torch
from typing import Dict, List
from torch.utils.data import DataLoader

from src.data.dataset import NormalizationDataset, collate_fn
from src.utils.logging_utils import setup_logger

logger = setup_logger("tahimik.efficiency")


class EfficiencyBenchmark:
    """
    Benchmarks inference time and GPU memory for a trained model.

    The protocol follows the manuscript exactly:
        1. 5 CUDA warmup passes (results discarded)
        2. 20 timed inference runs
        3. Report mean and std of per-sentence inference time
        4. Report peak GPU memory across all runs

    Args:
        model: A trained TAHIMIK model variant.
        tokenizer: The ByT5 tokenizer.
        device: torch.device to benchmark on.
        num_beams: Beam search width for generation.
        warmup_passes: Number of warmup passes before timing.
        inference_runs: Number of timed measurement runs.
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
    def _run_inference(
        self,
        dataloader: DataLoader,
    ) -> float:
        """Run one full inference pass and return total time in seconds."""
        self.model.eval()
        total_time = 0.0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=1024,
                num_beams=self.num_beams,
            )

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            total_time += time.perf_counter() - start

        return total_time

    def benchmark(
        self,
        test_dataset: NormalizationDataset,
        batch_size: int = 1,
    ) -> Dict[str, float]:
        """
        Run the full efficiency benchmark.

        Uses batch_size=1 by default to measure per-sentence time
        as described in the manuscript.

        Args:
            test_dataset: The test split NormalizationDataset.
            batch_size: Batch size for inference.

        Returns:
            Dict with:
                'avg_time_per_sentence': mean inference time (seconds)
                'std_time_per_sentence': std of per-run averages
                'peak_gpu_memory_mb': peak GPU memory in MB
                'num_sentences': number of test sentences
        """
        dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        num_sentences = len(test_dataset)

        # ── Warmup passes ───────────────────────────────────────────────
        logger.info(f"Running {self.warmup_passes} warmup passes...")
        for _ in range(self.warmup_passes):
            self._run_inference(dataloader)

        # Reset memory stats after warmup
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        # ── Timed runs ──────────────────────────────────────────────────
        logger.info(f"Running {self.inference_runs} timed inference passes...")
        per_run_times: List[float] = []

        for run in range(self.inference_runs):
            run_time = self._run_inference(dataloader)
            per_sentence_time = run_time / max(num_sentences, 1)
            per_run_times.append(per_sentence_time)

        # ── Compute statistics ──────────────────────────────────────────
        times_tensor = torch.tensor(per_run_times)
        avg_time = times_tensor.mean().item()
        std_time = times_tensor.std().item()

        # Peak GPU memory
        if self.device.type == "cuda":
            peak_memory_bytes = torch.cuda.max_memory_allocated(self.device)
            peak_memory_mb = peak_memory_bytes / (1024 * 1024)
        else:
            peak_memory_mb = 0.0

        results = {
            "avg_time_per_sentence": avg_time,
            "std_time_per_sentence": std_time,
            "peak_gpu_memory_mb": peak_memory_mb,
            "num_sentences": num_sentences,
        }

        logger.info(f"Efficiency results:")
        logger.info(f"  Avg time/sentence: {avg_time*1000:.2f} ms (+/- {std_time*1000:.2f})")
        logger.info(f"  Peak GPU memory:   {peak_memory_mb:.1f} MB")

        return results
