import time
import torch
from typing import Dict, List
from torch.utils.data import DataLoader
from src.data.dataset import NormalizationDataset, collate_fn

class EfficiencyBenchmark:
    def __init__(self, model, tokenizer, device: torch.device, num_beams=4, warmup_passes=5, inference_runs=20):
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.num_beams, self.warmup_passes, self.inference_runs = num_beams, warmup_passes, inference_runs

    @torch.no_grad()
    def _run_inference(self, dataloader):
        self.model.eval(); durations=[]
        for batch in dataloader:
            ids, mask = batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device)
            if self.device.type == "cuda": torch.cuda.synchronize()
            start=time.perf_counter(); self.model.generate(input_ids=ids, attention_mask=mask, max_length=1024, num_beams=self.num_beams)
            if self.device.type == "cuda": torch.cuda.synchronize()
            durations.append(time.perf_counter()-start)
        return durations

    def benchmark(self, test_dataset: NormalizationDataset, batch_size=1) -> Dict[str, object]:
        if batch_size != 1: raise ValueError("Chapter 3 per-sentence latency requires batch_size=1")
        loader=DataLoader(test_dataset,batch_size=1,shuffle=False,collate_fn=collate_fn); n=len(test_dataset)
        for _ in range(self.warmup_passes): self._run_inference(loader)
        timed=[]; memory_runs=[]
        for _ in range(self.inference_runs):
            if self.device.type == "cuda": torch.cuda.reset_peak_memory_stats(self.device)
            durations=self._run_inference(loader); timed.append(durations)
            if self.device.type == "cuda": memory_runs.append(torch.cuda.max_memory_allocated(self.device)/(1024*1024))
        per_sentence=[sum(row[i] for row in timed)/len(timed) for i in range(n)] if n else []
        run_means=[sum(row)/max(len(row),1) for row in timed]
        return {"avg_time_per_sentence": float(sum(run_means)/max(len(run_means),1)), "std_time_per_sentence": float(torch.tensor(run_means).std().item()) if len(run_means)>1 else 0.0, "per_sentence_time_seconds": per_sentence, "run_mean_time_seconds": run_means, "peak_gpu_memory_mb": (max(memory_runs) if memory_runs else None), "peak_gpu_memory_runs_mb": memory_runs, "gpu_memory_available": bool(memory_runs), "num_sentences": n}
