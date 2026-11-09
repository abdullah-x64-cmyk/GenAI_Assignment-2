# Task 6 browser verification

Observed 10 November 2026 in Chrome at http://127.0.0.1:7860.
The actual fine-tuned Qwen2.5-0.5B-Instruct model ran locally on CPU using
greedy decoding with KV caching. No benchmark test-set cases were used.

Synthetic patient entered through the browser:

```
1. The patient is a 58-year-old adult with lung cancer.
2. ECOG performance status is 1.
3. The patient has no history of heart failure.
4. Kidney function has not been documented.
```

| Criterion type | Criterion | Model label | Cited IDs | Generation time |
|---|---|---|---|---|
| Inclusion | ECOG performance status must be 0 or 1. | included | 1 | 10.70 s |
| Exclusion | History of heart failure. | not excluded | 3 | 7.47 s |

The browser displayed both predictions and the exact original text for each
cited ID. The inclusion citation is an error: sentence 1 does not establish
ECOG status; sentence 2 is relevant. The exclusion citation correctly quotes
the explicitly documented absence of heart failure. These observations are
a functional demonstration, not a new estimate of model accuracy.

The Inference lab displayed the saved T4 cache/no-cache and speculative/greedy
comparisons, including timings, throughput, peak memory, and token equality.
The Download report button produced a readable HTML file. Its contents were
inspected and copied unchanged to `task6_demo_report.html`, including the
prototype notice and the model's evidence-selection error. The Print control
was not separately exercised in this run.

The initial model download required a retry. The completed 988,097,824-byte
model file was checked against its published SHA-256 hash:
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`.
The app now defaults to the standard HTTP downloader for local model files.
