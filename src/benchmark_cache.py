"""Validate greedy KV-cache equivalence and benchmark latency/memory."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from cache_decode import cached_greedy, uncached_greedy
from common import prompt_for


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def peak_memory(device):
    return int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None


def run_once(fn, model, ids, output_tokens, eos, device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    result = fn(model, ids, output_tokens, eos)
    peak = peak_memory(device)
    result_dict = {
        "prompt_tokens": result.prompt_tokens,
        "generated_tokens": len(result.generated_token_ids),
        "prefill_seconds": result.prefill_seconds,
        "decode_seconds": result.decode_seconds,
        "tokens_per_second": result.tokens_per_second,
        "peak_memory_bytes": peak,
        "peak_incremental_memory_bytes": max(0, peak - baseline) if peak is not None else None,
        "generated_token_ids": result.generated_token_ids,
        "cache_sequence_lengths": result.cache_sequence_lengths,
    }
    return result_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen_lora.json")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--mode", choices=["base", "finetuned"], default="finetuned")
    parser.add_argument("--input-lengths", default="64,256,512")
    parser.add_argument("--output-lengths", default="16,32")
    parser.add_argument("--out", default="results/cache_benchmark.json")
    parser.add_argument("--include-token-ids", action="store_true")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16", "auto"],
                        default="float32", help="float32 reduces precision-induced argmax drift")
    parser.add_argument("--require-exact-match", action="store_true",
                        help="exit with an error if cached and reference greedy tokens differ")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    # Use the same eager attention implementation for both paths. This avoids
    # comparing distinct SDPA kernels (full query versus one-token query) and
    # reduces numerical variation during the token-equivalence check.
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16, "auto": "auto"}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"], torch_dtype=dtype, attn_implementation="eager",
    )
    if args.mode == "finetuned":
        model = PeftModel.from_pretrained(model, f"{config['output_dir']}/best_adapter")
    model.eval()
    # This benchmark intentionally uses one device so peak CUDA allocation and
    # cache placement are directly comparable between decoding paths.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if device.type == "meta":
        raise RuntimeError("model parameters are on meta; load the model on a concrete device")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    rows = read_rows(f"data/processed/{args.split}.jsonl")
    if not rows:
        raise ValueError(f"no rows found for {args.split}")
    base_ids = tokenizer(prompt_for(rows[0]), return_tensors="pt", truncation=True,
                         max_length=max(map(int, args.input_lengths.split(","))))["input_ids"][0]
    if len(base_ids) == 0:
        raise ValueError("tokenized benchmark prompt is empty")
    input_lengths = [int(x) for x in args.input_lengths.split(",")]
    output_lengths = [int(x) for x in args.output_lengths.split(",")]
    records = []
    for input_length in input_lengths:
        # Construct exact-length prompts from the same real validation prompt;
        # repeat its token sequence only when a requested length exceeds it.
        ids = base_ids.repeat((input_length + len(base_ids) - 1) // len(base_ids))[:input_length]
        ids = ids.unsqueeze(0).to(device)
        for output_length in output_lengths:
            # Warm both execution shapes once so CUDA context/kernel startup is
            # not charged to the first timed prefill.
            warmup_tokens = min(output_length, 4)
            cached_greedy(model, ids, warmup_tokens, tokenizer.eos_token_id)
            uncached_greedy(model, ids, warmup_tokens, tokenizer.eos_token_id)
            cached = run_once(cached_greedy, model, ids, output_length, tokenizer.eos_token_id, device)
            uncached = run_once(uncached_greedy, model, ids, output_length, tokenizer.eos_token_id, device)
            match_len = min(len(cached["generated_token_ids"]), len(uncached["generated_token_ids"]))
            agreement = sum(cached["generated_token_ids"][i] == uncached["generated_token_ids"][i]
                            for i in range(match_len))
            token_agreement = agreement / max(len(cached["generated_token_ids"]), len(uncached["generated_token_ids"]), 1)
            mismatch_indices = [i for i in range(match_len)
                                if cached["generated_token_ids"][i] != uncached["generated_token_ids"][i]]
            mismatch_indices.extend(range(match_len, max(len(cached["generated_token_ids"]),
                                                           len(uncached["generated_token_ids"]))))
            record = {
                "input_length": input_length,
                "requested_output_length": output_length,
                "cache": cached,
                "no_cache_reference": uncached,
                "token_exact_match": cached["generated_token_ids"] == uncached["generated_token_ids"],
                "token_agreement_fraction": token_agreement,
                "mismatch_token_indices": mismatch_indices,
            }
            if not args.include_token_ids:
                del record["cache"]["generated_token_ids"]
                del record["no_cache_reference"]["generated_token_ids"]
            records.append(record)
            print(json.dumps({k: v for k, v in record.items() if k not in ("cache", "no_cache_reference")}
                             | {"cache_decode_s": cached["decode_seconds"],
                                "no_cache_decode_s": uncached["decode_seconds"],
                                "cache_tok_s": cached["tokens_per_second"],
                                "no_cache_tok_s": uncached["tokens_per_second"]}))

    output = {
        "model": config["model_name"], "mode": args.mode, "split_prompt_source": args.split,
        "device": str(device), "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "dtype": str(next(model.parameters()).dtype),
        "warmup_tokens_per_path": min(4, min(output_lengths)) if output_lengths else 0,
        "attention_implementation": "eager",
        "method": "cached: one prompt prefill, then one-token forward calls carrying past_key_values; reference: full-prefix forward at every output step",
        "records": records,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved {len(records)} benchmark cases to {out_path}")
    if args.require_exact_match and any(not item["token_exact_match"] for item in records):
        raise SystemExit("cached and no-cache greedy sequences differed; inspect benchmark JSON and mismatch indices")


if __name__ == "__main__":
    main()
