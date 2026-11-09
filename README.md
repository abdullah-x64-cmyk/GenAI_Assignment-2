# TrialGPT Criterion Eligibility System (Tasks 1 and 2)

This project uses the required NLM TrialGPT Criterion Annotations dataset. It creates the fixed patient-grouped split, excludes all GPT-4 generated fields from model input, fine-tunes an open-weight decoder model, and provides criterion-level prediction, evidence extraction, and evaluation for inclusion and exclusion criteria.

## Chosen model and feasibility

Primary model: `Qwen/Qwen2.5-0.5B-Instruct` (0.49B parameters, decoder-only causal Transformer, Apache-2.0, Qwen2 tokenizer, model-card context up to 32,768 tokens). FP16/BF16 weights require about 1 GB, while 4-bit weights require roughly 0.3–0.5 GB before runtime/cache overhead. LoRA training with sequence length 2048, microbatch 1, gradient accumulation 8, and gradient checkpointing is practical on a free 8–16 GB GPU; CPU training is technically possible but slow. Training and evaluation were run in Google Colab on a Tesla T4 GPU. The adapter, training history, split manifest, truncation records, and held-out base/fine-tuned prediction logs have been downloaded to `results/task1_submission_artifacts.zip`.

Candidate: `HuggingFaceTB/SmolLM2-360M-Instruct` (0.36B parameters, decoder-only Llama-style causal Transformer, Apache-2.0, 8K context per model card). Its smaller weights reduce memory, but Qwen's longer context and stronger multilingual/instruction-tuning profile make Qwen preferable for patient-note conditioning. Both have compatible open-weight causal-generation architectures. Exact model metadata and license should be rechecked at download time.

## Reproduce

Use Python 3.10–3.12 and a free Kaggle/Colab GPU (or local CUDA). Install `requirements.txt`; then:

```bash
python src/prepare_data.py --out data/processed
python src/train.py --config configs/qwen_lora.json
python src/evaluate.py --config configs/qwen_lora.json --split validation
python src/evaluate.py --config configs/qwen_lora.json --split test
```

Run the final test evaluation only after selecting hyperparameters from validation. Artifacts go to `outputs/`; the current adapter and held-out logs are in the linked archive. The standalone split manifest is `results/split_manifest.json`.

## Data and leakage controls

The source is `ncbi/TrialGPT-Criterion-Annotations` from Hugging Face. `prepare_data.py` sorts unique patient IDs, shuffles with `numpy.random.default_rng(42)`, and allocates 37/8/8 patients. It saves patient IDs and row counts and asserts disjoint groups. Inputs use only `note`, `criterion_type`, and `criterion_text`; targets use `expert_eligibility` and `expert_sentences`. The columns `gpt4_explanation`, `gpt4_sentences`, `gpt4_eligibility`, and `explanation_correctness` are explicitly excluded.

Patient notes already contain numbered sentences (for example `0. ... 1. ...`). The output must be JSON with `eligibility` in the source expert label vocabulary and `evidence_sentence_ids` as integer IDs drawn from that note. Empty evidence is allowed. Evidence IDs are learned as target text; invalid IDs are counted in evaluation.

## Truncation

Tokenization uses the model's native tokenizer and truncates inputs to 2048 tokens by default, retaining the beginning of the concatenated prompt. This can remove later clinical evidence; the script records tokenized input length, truncation flag, and source note length per example. For the report, include the fraction truncated and inspect those cases. Do not silently claim evidence completeness for truncated notes. Increase `max_length` if free GPU memory permits.

## Experimental design

Supervised objective: causal next-token cross-entropy on the structured target JSON only; prompt tokens receive label `-100`. LoRA rank 8, alpha 16, dropout 0.05, target modules `q_proj`, `k_proj`, `v_proj`, `o_proj`; AdamW learning rate 2e-4, 3 epochs, effective batch 8, warmup ratio 0.05, seed 42. Save the best checkpoint by validation loss. Greedy decoding (`do_sample=False`, temperature omitted) uses the same prompt/template and max 128 new tokens for base and tuned models.

Evaluation includes accuracy, macro-F1, per-class precision/recall/F1, and confusion matrices overall and separately for inclusion and exclusion criteria. Evidence metrics use exact sentence-ID sets. If the entire evaluated group has empty gold and predicted evidence, precision/recall/F1 are 1; if only one side is empty, the affected metric is 0. Test metrics must be generated only once after configuration is fixed. The checkpoint is selected by validation loss.

## Task 2: Criterion-level eligibility and evidence

`src/evaluate.py` returns one record per criterion with its criterion text and type, the predicted expert category, sentence IDs, verbatim text for IDs that match numbered note sentences, invalid/unmatched IDs, and a deterministic audit explanation. The explanation quotes only the cited note text; if no evidence is cited, it explicitly says that missing evidence does not prove eligibility or ineligibility. The model is asked to preserve categories such as `included`, `not included`, `excluded`, `not excluded`, `not enough information`, and `not applicable` as distinct outputs. Unsupported or type-incompatible categories are retained in metrics and flagged rather than silently rewritten.

Run `python src/evaluate.py --config configs/qwen_lora.json --split validation --mode finetuned` to generate the enriched per-criterion output on validation. The evaluator stores overall and inclusion/exclusion metrics, per-class reports, confusion matrices, exact evidence metrics, and error cases in `outputs/evaluation/`. The final test split has already been evaluated; `src/analyze_saved_evaluations.py` re-aggregates its existing base and fine-tuned logs without rerunning model inference:

```bash
python src/analyze_saved_evaluations.py --archive results/task1_submission_artifacts.zip --split test --out results/task2_error_analysis.json
```

### Held-out Task 2 analysis

The analysis uses the existing 206-criterion test logs (75 inclusion and 131 exclusion records); it does not tune or rerun the model on test data. The fine-tuned model achieved 0.640 inclusion accuracy / 0.339 macro-F1 and 0.687 exclusion accuracy / 0.271 macro-F1. Evidence F1 was 0.576 for inclusion and 0.190 for exclusion, indicating weaker evidence retrieval on exclusion criteria. Full per-class precision/recall/F1 and confusion matrices for both model modes are in [`results/task2_error_analysis.json`](results/task2_error_analysis.json).

On the 66 gold `not enough information` records, the fine-tuned system preserved that category for 31 and made a definitive prediction for 35; notably, all 28 exclusion cases with that gold label were predicted `not excluded`. The model cited no evidence for 26 of those 28 cases. It also produced two opposing definitive inclusion errors (`not included` predicted as `included`), including patient `sigir-201410`, where no evidence was annotated but sentence IDs 0 and 1 were returned. `not applicable` had zero F1 (15 test cases), so the model did not reliably preserve that category in its predictions. These are errors from the model/gold comparison; they do not establish contradictions in the underlying medical record. Each per-case record and representative examples are included in the analysis JSON.

The existing Task 1 prediction logs predate the richer output schema and therefore cannot provide extracted evidence text for those historical predictions. The current evaluator adds the criterion, verified evidence text, unmatched-ID flags, and evidence-grounded audit explanation to each newly generated record. The historical test predictions were re-aggregated only; no new test inference was run for this task.

## Measured results

The fixed test split has 206 criteria from 8 held-out patients. Training ran for 3 epochs; its runtime was 622.75 seconds, final logged training loss was 0.1860, and validation loss was 0.1209. Metrics observed in Colab:

| Split / model | Accuracy | Macro-F1 | Evidence micro-F1 | Mean generation time |
|---|---:|---:|---:|---:|
| Validation, base | 0.000 | 0.000 | 0.0207 | 4.272 s/criterion |
| Validation, LoRA | 0.718 | 0.354 | 0.441 | 1.031 s/criterion |
| Test, base | 0.000 | 0.000 | 0.0432 | 3.594 s/criterion |
| Test, LoRA | 0.670 | 0.398 | 0.483 | 0.984 s/criterion |

Reported truncation fraction was 0.000 for both held-out test runs. The base model often emitted free-form or dictionary-like text rather than the requested JSON, so the evaluator assigned zero label accuracy and macro-F1. This baseline is primarily an output-schema failure and must be discussed with the scores. Test data was not used for tuning. These are screening research results, not a clinical enrollment decision.

### Exporting submission artifacts from Colab

Evaluation saves complete logs under `outputs/evaluation/` (predictions, class reports, confusion matrices, truncation flags, and latencies); training saves the adapter at `outputs/qwen2.5-0.5b-lora/best_adapter/` and training metrics at `outputs/qwen2.5-0.5b-lora/train_metrics.json`. The Colab artifacts are in [`results/task1_submission_artifacts.zip`](results/task1_submission_artifacts.zip), which includes the adapter, fine-tuned test predictions, aggregate base-model metrics, training metrics, truncation records, and split manifest. Unstructured base generations are omitted from the public bundle because they can reproduce patient-note text. The archive is available from GitHub at [Download the trained adapter and benchmark artifacts](https://github.com/abdullah-x64-cmyk/GenAI_Assignment-2/raw/refs/heads/main/results/task1_submission_artifacts.zip).

To export again from Colab, first sanitize the raw logs and then download the bundle from `/content/task1_project`:

```python
!python src/package_submission_artifacts.py --root /content/task1_project --out /content/task1_submission_artifacts.zip
from google.colab import files
files.download('/content/task1_submission_artifacts.zip')
```

The public archive contains evaluation outputs and patient IDs in the split manifest, but no patient note text. The application remains a research screening prototype, not a clinical enrollment decision tool.

## Research sources

See `report/references.bib` and `report/task1.tex`. Sources include the TrialGPT paper, Transformer architecture, Qwen and SmolLM2 model cards/papers, LoRA, and FlashAttention. The report and `results/benchmark_summary.json` describe the observed validation and held-out results; the archive contains the detailed test logs and trained adapter.

## Task 3: Explicit KV-cached autoregressive decoding

`src/cache_decode.py` implements two deterministic greedy paths. `cached_greedy`
prefills the unpadded prompt once, then feeds each newly generated token with
the returned `past_key_values` on the next forward call. The key/value tensors
therefore grow by one position per decode call. `uncached_greedy` is the
reference: it appends each generated token to the prefix and reruns the model
over the complete prefix with `use_cache=False`. Both paths share argmax token
selection and EOS handling.

For Qwen2.5, prompt positions use the model's zero-based rotary positions. The
cached path explicitly supplies the next position as `prompt_length + step`;
the model's causal mask prevents queries from attending to future positions,
and the full attention mask grows by one visible token with the KV cache. There
are no padding positions in these benchmark prompts. Floating-point kernel
ordering can change an argmax for nearly tied logits, so the benchmark records
both exact-sequence match and token agreement.
The benchmark fixes Transformers to eager attention for both cached and
reference forwards so it compares the same attention implementation.

Run in the same GPU environment used for fine-tuning, after restoring model and
adapter artifacts:

```bash
python src/benchmark_cache.py --config configs/qwen_lora.json --split validation --mode finetuned --dtype float32 --input-lengths 64,256,512 --output-lengths 16,32 --out results/cache_benchmark.json
```

The script checks actual cache sequence length after prefill and every
incremental update, and reports those lengths alongside token agreement,
prefill latency, decode latency, generated tokens per second, and total plus
incremental peak CUDA allocation for both methods. Add
`--require-exact-match` to make the run fail if any output token differs. Use
the same device, dtype, model, adapter, and prompts for a fair comparison. A
CPU run can check token agreement and control flow, but cannot supply CUDA peak
memory and is not a substitute for the assignment's requested GPU comparison.
The final measured run is saved in `results/cache_benchmark_final.json`. It used
the fine-tuned Qwen2.5-0.5B adapter, a Tesla T4, float32, eager attention for
both paths, and four warmup tokens per path. All six cached and uncached output
sequences matched exactly. The earlier bfloat16/default-backend run diverged
for 64- and 256-token prompts; changing both the backend to explicit eager and
the computation to float32 removed the observed divergences. Since both
settings changed together, this comparison does not isolate which change was
responsible. The final measurements (times in seconds, throughput in generated
tokens/s, peak memory in GiB) are:

| Prompt / requested output | Exact match | Prefill cached / reference | Decode cached / reference | Throughput cached / reference | Peak memory cached / reference |
|---:|:---:|---:|---:|---:|---:|
| 64 / 16 | Yes | 0.055 / 0.046 | 0.662 / 0.746 | 24.18 / 21.46 | 1.89 / 1.94 |
| 64 / 32 | Yes | 0.050 / 0.066 | 1.385 / 1.523 | 23.10 / 21.01 | 1.89 / 1.96 |
| 256 / 16 | Yes | 0.071 / 0.066 | 0.258 / 0.581 | 27.14 / 12.05 | 2.01 / 2.15 |
| 256 / 32 | Yes | 0.074 / 0.067 | 0.260 / 0.585 | 26.93 / 11.96 | 2.01 / 2.15 |
| 512 / 16 | Yes | 0.154 / 0.147 | 0.880 / 2.637 | 18.19 / 6.07 | 2.16 / 2.45 |
| 512 / 32 | Yes | 0.151 / 0.149 | 1.372 / 5.495 | 23.33 / 5.82 | 2.16 / 2.47 |

Caching lowered measured decode time and peak allocation in all six cases.
The speed advantage grew with prompt length: for 512-token prompts, cached
throughput was about 3.0--4.0 times higher. The benchmark is small and uses a
single validation prompt per requested length, so these timings describe this
T4 run rather than a general hardware guarantee. The old exploratory results
remain in `results/cache_benchmark_preliminary.csv` for provenance; use the
final JSON log for Task 3 conclusions.

## Task 4: Tiled attention with online softmax

`src/online_attention.py` provides a materialized scaled dot-product attention
reference and `tiled_online_softmax_attention`. The tiled method iterates over
query and key/value blocks and keeps a running row maximum, softmax
normalization sum, and weighted-value accumulator. It rescales the prior sum
and accumulator whenever a new tile changes the maximum, which is the online
log-sum-exp update used to avoid retaining all sequence scores at once. Boolean
masks use `True` for visible positions; floating masks add score bias (use
negative infinity to mask). Causal masking supports a query suffix into a
longer key/value prefix using bottom-right alignment. Half and bfloat16 inputs
are upcast to float32 for softmax accumulation, and outputs are cast back.

Run the correctness and timing harness on CPU or CUDA:

```bash
python -m unittest discover -s tests -v
python src/benchmark_attention.py --device auto --lengths 64,256 --batch 1 --heads 4 --head-dim 32 --include-sdpa --out results/attention_benchmark.json
```

Correctness tests compare small controlled causal, masked, suffix-query, and
uneven-tile cases against the reference and PyTorch SDPA using tight float64
tolerances. The current local CPU microbenchmark is in
`results/attention_benchmark_cpu.json` (B=1, H=2, D=16, float32, two timed
iterations). At lengths 32 and 64, the reference took 2.16 and 0.76 ms, the
educational tiled implementation 3.68 and 7.69 ms, and PyTorch SDPA 0.15 ms
for both. These small CPU timings are noisy and do not predict GPU performance.

A separate Colab Tesla T4 run used B=1, H=4, D=32, float32, causal masking,
tile sizes 32x64, three warmups, and ten timed iterations. The full measured
summary is in `results/attention_benchmark_t4.json`. All methods agreed with
the reference under the configured tolerance at lengths 64, 256, and 512
(maximum absolute error for tiled attention was below 4.2e-7). Median times
(reference / tiled / PyTorch SDPA, ms) were 0.744 / 1.773 / 0.070 at length
64, 0.727 / 22.877 / 0.140 at 256, and 0.900 / 89.215 / 0.250 at 512. The
educational tiled implementation reduced measured peak CUDA allocation at
length 512 from 30.83 MB to 10.38 MB, while increasing runtime substantially.
CUDA peak allocation includes live tensors and allocator/runtime effects, not
just the score matrix. These results cover one GPU run and should not be
interpreted as broad hardware performance guarantees.

The tiled method is for demonstrating the algorithm and controlling
intermediate score storage. It uses Python loops and multiple ordinary PyTorch
operators per tile, so it cannot fuse loads, softmax updates, and value
accumulation into one GPU kernel. A hardware-optimized fused attention kernel
can therefore be faster despite doing the same stable computation.

## Task 5: Greedy speculative decoding

`src/speculative_decode.py` implements candidate drafting, block verification
by the target, greedy-prefix acceptance, rejection correction, and cache
rollback/resynchronization. The draft is constructed from the target's merged
Qwen2.5-0.5B checkpoint by retaining its first two decoder layers and sharing
its embeddings, final norm, and output head. It uses the target tokenizer and
vocabulary exactly, and inherits the target's fine-tuned weights in the
retained layers. This gives a smaller draft without token-ID conversion; it
may still have low acceptance because it omits most target layers. If every
draft token matches, the target's final verification logits provide one
additional greedy token.

Run the toy correctness cases and the real benchmark as follows:

```bash
python -m unittest tests.test_speculative_decode -v
python src/benchmark_speculative.py --config configs/qwen_lora.json --split validation --mode finetuned --lengths 128,512 --max-new-tokens 64 --draft-tokens 4 --draft-layers 2 --out results/speculative_benchmark.json
```

The benchmark compares against cached ordinary greedy on the same validation
prompt prefixes and checks exact generated token IDs. It records prefill,
decode, total latency, throughput, draft acceptance, cache lengths, and CUDA
peak/incremental allocation. A T4 run with the fine-tuned Qwen target, eager
attention, float32, four draft tokens per round, and 64 generated tokens is
saved in `results/speculative_benchmark_t4.json`. Both cases matched ordinary
greedy token for token. The two-layer draft accepted 0 of 250 proposals in
each case (0% acceptance), so speculative decoding was slower: for 128 input
tokens, total latency was 10.170 s versus 2.076 s and total throughput was
6.29 versus 30.82 tokens/s; for 512 input tokens, total latency was 7.721 s
versus 2.166 s and total throughput was 8.29 versus 29.54 tokens/s. Peak CUDA
allocation was 2166.92 versus 2085.61 MiB and 2624.08 versus 2320.78 MiB,
respectively. This is an expected failure mode for a weak draft: the target
must still generate every correction, while the draft adds serial forward
passes and cache storage. The benchmark uses validation prompts only and is
one hardware run. CPU toy tests verify acceptance, rejection correction, EOS,
and cache alignment; the T4 token equality check provides model-level output
correctness evidence, not a speedup claim.

## Task 6: Browser-based screening application

The local application is served by `src/web_app.py`; its browser interface is
in `web/`. On Windows, double-click `setup_and_run.bat`. The first launch
creates a local virtual environment and installs `requirements-app.txt`, then
opens `http://127.0.0.1:7860` in the browser. Keep the terminal window open
while using the application; closing it stops the local server. The first
screening run loads Qwen2.5-0.5B-Instruct and the LoRA adapter. The adapter is
restored from `results/task1_submission_artifacts.zip` if needed, and the base
model is downloaded from Hugging Face on first use. Inference runs on a
detected CUDA GPU or falls back to CPU. No paid API or model service is used.

Enter a patient record and add one or more inclusion or exclusion criteria.
Number the patient-note sentences (for example, `1. ... 2. ...`) so the
application can return the exact quoted evidence text for sentence IDs. The
application preserves the model's criterion-level label, including “not
enough information” and “not applicable”; absent evidence is never converted
into an eligibility decision. Results include the criterion, label, exact
matching note sentences, explanation, generation time, selected decoding
path, and any invalid citation IDs. A prompt exceeding the configured input
window is left-truncated to retain the criterion and later text; the result
shows a warning and token counts. The report can be downloaded as readable
HTML or printed from the browser. A prominent prototype notice appears in
the interface and exported report.

The **Inference lab** view reads the saved project benchmark logs and shows
the Task 3 cache/no-cache comparison and Task 5 speculative/ordinary greedy
comparison, including token equality, times, throughput, and acceptance
results. These are measured T4 validation-run results; they are clearly
identified as hardware-specific. Users can also choose cached greedy or the
custom two-layer speculative path for a screening run. Speculative decoding
may be slower, as the measured benchmark demonstrates.

The server binds to `127.0.0.1` by default. To use another local port, set
`TRIAL_APP_PORT` before launching; `TRIAL_APP_NO_BROWSER=1` disables automatic
browser opening. The server does not write request bodies or patient records
to disk or access logs. Screening results exist in the browser until exported.
