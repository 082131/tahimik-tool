# SOP 2: Computational-Efficiency Metrics

SOP 2 asks how much computational time and GPU memory each model requires during inference. It does not measure whether the generated text is correct; that is SOP 1.

The two metrics are:

1. Average inference time per sentence
2. Peak GPU memory during inference

## 1. Average inference time per sentence

### What it measures

This measures how long a model takes to generate a normalized output for one input sentence. A lower value means faster inference.

The measurement is taken after the model and tokenizer have already been loaded. It measures the generation call, including the model's encoder, delete gate (when present), decoder, and beam-search generation.

### How the code measures it

The efficiency benchmark follows this sequence:

1. Put the model in evaluation mode and disable gradients.
2. Use batch size 1, so each timing corresponds to one sentence.
3. Run five warm-up passes. These initialize CUDA contexts, kernels, and memory pools; warm-up times are not included in the result.
4. Run 20 timed passes over the same test sentences.
5. Synchronize CUDA immediately before and after `model.generate()`. GPU operations are asynchronous, so synchronization is required to measure actual execution time.
6. Store the elapsed time for every sentence in every timed pass.
7. Average the repeated timings for each sentence.
8. Average the sentence timings to obtain the overall average inference time.

### Formula

If sentence $i$ is timed in $R$ repeated runs, its average time is:

$$
\bar{t}_i=\frac{1}{R}\sum_{r=1}^{R}t_{i,r}
$$

Each symbol has a specific role:

| Symbol | Meaning in the implementation |
|---|---|
| $i$ | Index of one test sentence, from 1 through $N$. |
| $r$ | Index of one timed repetition, from 1 through $R$. |
| $t_{i,r}$ | Elapsed generation time for sentence $i$ during run $r$, measured in seconds. |
| $R$ | Number of timed repetitions; the current configuration uses 20. |
| $\sum_{r=1}^{R}$ | Adds the 20 timings recorded for the same sentence. |
| $\frac{1}{R}$ | Divides that sum by 20 to obtain the sentence's average time. |
| $\bar{t}_i$ | Average time for sentence $i$ across the timed repetitions. |

If there are $N$ test sentences, the dataset average is:

$$
\text{Average inference time per sentence}
=\frac{1}{N}\sum_{i=1}^{N}\bar{t}_i
$$

For this second formula:

| Symbol | Meaning |
|---|---|
| $N$ | Number of sentences in the test set. |
| $i=1$ | Start with the first sentence. |
| $i=N$ | End with the last sentence. |
| $\sum_{i=1}^{N}$ | Adds the average time of every test sentence. |
| $\frac{1}{N}$ | Divides the total by the number of sentences. |
| $\bar{t}_i$ | The already-averaged time for sentence $i$. |

In plain language: first average the repeated timings for each sentence, then average those sentence averages. The result answers, “On average, how long does this model take to normalize one sentence?”

### Exact mapping to the code

In `src/evaluation/efficiency.py`:

| Code operation | Role in the formula |
|---|---|
| `model.eval()` | Disables training behavior such as dropout. |
| `@torch.no_grad()` | Prevents gradient storage, so timing and memory represent inference rather than training. |
| `DataLoader(..., batch_size=1, shuffle=False)` | Presents one sentence at a time and keeps sentence order reproducible. |
| `time.perf_counter()` | High-resolution wall-clock timer. |
| `self.model.generate(...)` | The operation being timed: encoding, compression/gating when present, decoding, and beam search. |
| `durations.append(elapsed)` | Stores each $t_{i,r}$. |
| `timed_runs.append(durations)` | Stores all sentence timings from one complete run. |
| `per_sentence_time` list comprehension | Computes each $\bar{t}_i$ across runs. |
| `run_means` | Computes the mean latency of each complete run. |
| `avg_time_per_sentence` | Computes the final dataset average. |
| `std_time_per_sentence` | Standard deviation of the run means, reported as run-mean SD. |

CUDA synchronization occurs immediately before starting and immediately after `generate()` finishes. Without it, the CPU timer could stop before queued GPU work has actually completed.

The code also calculates the mean time of each complete run and reports the standard deviation of those run means. The main unit is seconds per sentence; milliseconds per sentence may be shown in tables by multiplying by 1,000.

The run-level mean and SD are calculated as:

$$
\bar{t}_r=\frac{1}{N}\sum_{i=1}^{N}t_{i,r}
$$

$$
SD_{\text{run means}}
=\sqrt{\frac{1}{R-1}\sum_{r=1}^{R}(\bar{t}_r-\bar{t})^2}
$$

Here, $\bar{t}_r$ is one complete-run average and $\bar{t}$ is the average of the run means. This SD describes run-to-run timing stability; it is not the spread of individual sentence lengths.

### Small example

Suppose three sentences have average times:

```text
Sentence 1: 0.20 seconds
Sentence 2: 0.30 seconds
Sentence 3: 0.25 seconds
```

Then:

$$
\text{Average} = \frac{0.20+0.30+0.25}{3}=0.25\text{ seconds per sentence}
$$

The reported average is 250 milliseconds per sentence.

### What is not included

With the current implementation, model loading, tokenizer loading, dataset construction, and input transfer before the timer starts are not part of the generation-time measurement. This is appropriate when SOP 2 is intended to compare model inference itself. The same scope must be used for every model.

### Important controls

Keep these fixed across ByT5, MrT5, and TAHIMIK:

- GPU and software environment
- precision and evaluation mode
- batch size (the code requires 1)
- beam count (`num_beams=4` in the benchmark)
- maximum generation length
- warm-up count and number of timed runs
- test-sentence order and test set

## 2. Peak GPU memory during inference

### What it measures

Peak GPU memory is the largest amount of GPU memory allocated by the model during one complete inference pass. A lower value means the model requires less GPU memory.

This is not the model's file size and not the memory remaining on the GPU. It is the maximum memory allocated during the measured generation pass.

### How the code measures it

For each timed run:

1. Reset CUDA's peak-memory counter.
2. Run inference over the complete test set.
3. Read `torch.cuda.max_memory_allocated()` after the run.
4. Convert bytes to megabytes.
5. Store that run-level peak in the memory list.

The final headline value is the maximum of the recorded run-level peaks:

$$
\text{Peak GPU memory}
=\max_{r=1,\ldots,R}m_r
$$

where $m_r$ is the peak allocated memory during run $r$.

The memory formula's elements are:

| Symbol | Meaning |
|---|---|
| $R$ | Number of timed inference runs. |
| $m_r$ | Largest CUDA memory allocation observed during run $r$. |
| $\max$ | Selects the largest run-level peak. |
| `torch.cuda.reset_peak_memory_stats()` | Starts a fresh peak counter before each timed run. |
| `torch.cuda.max_memory_allocated()` | Reads the largest allocated CUDA memory value reached during that run. |
| $1024\times1024$ | Converts bytes to megabytes in the code. |

Peak memory includes allocations made while the batch is moved to the GPU and while `generate()` runs, because those operations occur after the per-run memory counter is reset. It does not include memory from the discarded warm-up measurements in the reported run list.

### Example

```text
Run 1: 2,410 MB
Run 2: 2,430 MB
Run 3: 2,421 MB
```

The reported peak is 2,430 MB, because that was the largest observed allocation. Reporting the maximum is conservative: it describes the largest memory requirement encountered during the benchmark.

If the benchmark runs on CPU, no CUDA memory value is available. The code returns `gpu_memory_available = false` and does not produce a meaningful GPU-memory comparison.

## How SOP 2 results should be presented

Report the average time per sentence with its run-level standard deviation, and report peak GPU memory in MB. Also preserve the underlying per-sentence latency list and per-run memory list; SOP 4 uses those observations for significance testing.

Example table format:

| Model | Average inference time (ms/sentence) | Run-mean SD (ms) | Peak GPU memory (MB) |
|---|---:|---:|---:|
| ByT5 | 250.0 | 8.2 | 2,430 |
| MrT5 | 190.0 | 6.4 | 2,180 |
| TAHIMIK | 145.0 | 5.1 | 1,760 |

The averages and maxima summarize SOP 2. They are not themselves the complete input to SOP 4: the statistical tests need the repeated per-sentence latency observations and matched per-run memory observations.

## Complete code-to-result flow

```text
Load model and test set
        ↓
Warm-up passes (5, discarded)
        ↓
For each of 20 timed runs:
  reset CUDA peak counter
  generate each sentence with batch size 1
  save each sentence duration
  save that run's peak GPU allocation
        ↓
Average repeated timings for each sentence
        ↓
Average sentence means → avg_time_per_sentence
Compute SD of run means → std_time_per_sentence
Take maximum memory run → peak_gpu_memory_mb
        ↓
Preserve per-sentence times and per-run memory for SOP 4
```
