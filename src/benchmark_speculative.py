"""Compare hand-written greedy speculative decoding with ordinary greedy."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from cache_decode import cached_greedy
from common import prompt_for
from speculative_decode import speculative_greedy


def make_draft(target, layers: int):
    """Create a smaller, tokenizer-identical draft by retaining early layers.

    The draft inherits the target's merged weights in the retained layers,
    embeddings, normalization, and output projection. No alternate tokenizer
    or token remapping is involved.
    """
    base = target.merge_and_unload() if isinstance(target, PeftModel) else target
    if not hasattr(base, "model") or not hasattr(base.model, "layers"):
        raise TypeError("draft construction currently supports Qwen2-style decoder models")
    total = len(base.model.layers)
    if not 1 <= layers < total:
        raise ValueError(f"draft layer count must be between 1 and {total - 1}")
    draft = copy.deepcopy(base)
    draft.model.layers = nn.ModuleList(list(draft.model.layers[:layers]))
    # Reuse vocabulary-scale weights instead of storing duplicate copies;
    # the draft differs only in its short decoder stack.
    draft.model.embed_tokens = base.model.embed_tokens
    draft.model.norm = base.model.norm
    draft.lm_head = base.lm_head
    draft.config.num_hidden_layers = layers
    draft.model.config.num_hidden_layers = layers
    draft.eval()
    return base, draft


def measure(fn, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
    else:
        baseline = 0
    result = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = int(torch.cuda.max_memory_allocated(device))
    else:
        peak = None
    return result, peak, (max(0, peak - baseline) if peak is not None else None)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen_lora.json")
    ap.add_argument("--split", choices=("validation", "test"), default="validation")
    ap.add_argument("--mode", choices=("base", "finetuned"), default="finetuned")
    ap.add_argument("--lengths", default="128,512")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--draft-tokens", type=int, default=4)
    ap.add_argument("--draft-layers", type=int, default=2)
    ap.add_argument("--out", default="results/speculative_benchmark.json")
    args = ap.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    target = AutoModelForCausalLM.from_pretrained(
        config["model_name"], torch_dtype=torch.float32, attn_implementation="eager"
    )
    if args.mode == "finetuned":
        target = PeftModel.from_pretrained(target, f"{config['output_dir']}/best_adapter")
    target.eval().to(device)
    target, draft = make_draft(target, args.draft_layers)
    target.eval().to(device)
    draft.eval().to(device)
    rows = read_rows(f"data/processed/{args.split}.jsonl")
    if not rows:
        raise ValueError(f"no rows found for split {args.split}")
    source = tokenizer(prompt_for(rows[0]), return_tensors="pt", truncation=True,
                       max_length=max(map(int, args.lengths.split(",")))).input_ids[0]
    records = []
    for length in map(int, args.lengths.split(",")):
        ids = source.repeat((length + len(source) - 1) // len(source))[:length].unsqueeze(0).to(device)
        # Warm both paths outside measured results.
        cached_greedy(target, ids, min(4, args.max_new_tokens), tokenizer.eos_token_id)
        speculative_greedy(target, draft, ids, min(4, args.max_new_tokens),
                           args.draft_tokens, tokenizer.eos_token_id)
        ordinary, ordinary_peak, ordinary_incremental = measure(
            lambda: cached_greedy(target, ids, args.max_new_tokens, tokenizer.eos_token_id), device
        )
        speculative, spec_peak, spec_incremental = measure(
            lambda: speculative_greedy(target, draft, ids, args.max_new_tokens,
                                       args.draft_tokens, tokenizer.eos_token_id), device
        )
        records.append({
            "input_tokens": length,
            "requested_output_tokens": args.max_new_tokens,
            "ordinary": {
                "generated_tokens": len(ordinary.generated_token_ids),
                "token_ids": ordinary.generated_token_ids,
                "prefill_seconds": ordinary.prefill_seconds,
                "decode_seconds": ordinary.decode_seconds,
                "total_seconds": ordinary.prefill_seconds + ordinary.decode_seconds,
                "tokens_per_second": ordinary.tokens_per_second,
                "overall_tokens_per_second": len(ordinary.generated_token_ids) / (ordinary.prefill_seconds + ordinary.decode_seconds),
                "peak_cuda_allocated_bytes": ordinary_peak,
                "incremental_peak_cuda_bytes": ordinary_incremental,
            },
            "speculative": {
                "generated_tokens": len(speculative.generated_token_ids),
                "token_ids": speculative.generated_token_ids,
                "prefill_seconds": speculative.prefill_seconds,
                "decode_seconds": speculative.decode_seconds,
                "total_seconds": speculative.prefill_seconds + speculative.decode_seconds,
                "tokens_per_second": speculative.tokens_per_second,
                "overall_tokens_per_second": len(speculative.generated_token_ids) / (speculative.prefill_seconds + speculative.decode_seconds),
                "drafted_tokens": speculative.drafted_tokens,
                "accepted_draft_tokens": speculative.accepted_draft_tokens,
                "acceptance_rate": speculative.acceptance_rate,
                "rejected_rounds": speculative.rejected_rounds,
                "rounds": speculative.rounds,
                "peak_cuda_allocated_bytes": spec_peak,
                "incremental_peak_cuda_bytes": spec_incremental,
                "target_cache_lengths": speculative.target_cache_lengths,
                "draft_cache_lengths": speculative.draft_cache_lengths,
            },
            "token_exact_match": ordinary.generated_token_ids == speculative.generated_token_ids,
        })
        print(json.dumps({k: v for k, v in records[-1].items() if k not in ("ordinary", "speculative")} |
                         {"ordinary_seconds": ordinary.decode_seconds,
                          "speculative_seconds": speculative.decode_seconds,
                          "acceptance_rate": speculative.acceptance_rate,
                          "drafted": speculative.drafted_tokens,
                          "accepted": speculative.accepted_draft_tokens,
                          "exact_match": ordinary.generated_token_ids == speculative.generated_token_ids}))
    out = {
        "model": config["model_name"], "mode": args.mode,
        "draft_model": f"{config['model_name']} truncated to {args.draft_layers} decoder layers",
        "tokenizer": config["model_name"], "draft_tokens_per_round": args.draft_tokens,
        "device": str(device), "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "dtype": str(next(target.parameters()).dtype), "attention_implementation": "eager",
        "target_draft_state": "Both caches contain identical accepted token history after each completed round; rejection crops to accepted prefix then advances both with target correction.",
        "records": records,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved {len(records)} speculative cases to {path}")
    if any(not r["token_exact_match"] for r in records):
        raise SystemExit("speculative decoding did not match ordinary greedy decoding")


if __name__ == "__main__":
    main()
