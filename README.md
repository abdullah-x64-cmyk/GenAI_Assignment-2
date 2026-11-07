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
python src/benchmark_cache.py --config configs/qwen_lora.json --split validation --mode finetuned --input-lengths 64,256,512 --output-lengths 16,32 --out results/cache_benchmark.json
```

The script checks actual cache sequence length after prefill and every
incremental update, and reports those lengths alongside token agreement,
prefill latency, decode latency, generated tokens per second, and total plus
incremental peak CUDA allocation for both methods. Add
`--require-exact-match` to make the run fail if any output token differs. Use
the same device, dtype, model, adapter, and prompts for a fair comparison. A
CPU run can check token agreement and control flow, but cannot supply CUDA peak
memory and is not a substitute for the assignment's requested GPU comparison.
Preliminary Tesla T4 measurements from the fine-tuned Qwen2.5-0.5B model in
bfloat16 are recorded in `results/cache_benchmark_preliminary.csv`. This run
reported `attention_implementation=None`, so it is not the planned shared
eager-backend comparison. Exact token agreement held for both 512-token prompts
but not for the 64- or 256-token prompts; cached decoding used less peak memory
in all six cases. These results are diagnostic and do not establish greedy
token equivalence. A same-backend rerun and investigation of the first
divergent token remain necessary before Task 3 can be considered complete.
