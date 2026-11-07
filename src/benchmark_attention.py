"""Correctness and optional speed/memory benchmark for tiled online attention."""
from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from online_attention import (
    reference_scaled_dot_product_attention,
    tiled_online_softmax_attention,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(fn, q, k, v, device, warmup, iterations):
    for _ in range(warmup):
        fn(q, k, v, causal=True)
    _sync(device)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
    else:
        baseline = None
    samples = []
    output = None
    for _ in range(iterations):
        _sync(device)
        start = time.perf_counter()
        output = fn(q, k, v, causal=True)
        _sync(device)
        samples.append(time.perf_counter() - start)
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return output, samples, baseline, peak


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--lengths", default="64,256")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--tile-q", type=int, default=32)
    parser.add_argument("--tile-k", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--include-sdpa", action="store_true",
                        help="also compare PyTorch scaled_dot_product_attention")
    parser.add_argument("--out", default="results/attention_benchmark.json")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]

    def reference(q, k, v, causal):
        return reference_scaled_dot_product_attention(q, k, v, causal=causal)

    def tiled(q, k, v, causal):
        return tiled_online_softmax_attention(q, k, v, causal=causal,
                                              tile_q=args.tile_q, tile_k=args.tile_k)

    methods = {"reference": reference, "tiled_online": tiled}
    if args.include_sdpa:
        methods["pytorch_sdpa"] = lambda q, k, v, causal: F.scaled_dot_product_attention(
            q, k, v, is_causal=causal
        )

    records = []
    for length in map(int, args.lengths.split(",")):
        generator = torch.Generator(device="cpu").manual_seed(314159 + length)
        q = torch.randn((args.batch, args.heads, length, args.head_dim),
                        generator=generator, dtype=torch.float32).to(device=device, dtype=dtype)
        k = torch.randn((args.batch, args.heads, length, args.head_dim),
                        generator=generator, dtype=torch.float32).to(device=device, dtype=dtype)
        v = torch.randn((args.batch, args.heads, length, args.head_dim),
                        generator=generator, dtype=torch.float32).to(device=device, dtype=dtype)
        outputs = {}
        for name, fn in methods.items():
            output, samples, baseline, peak = _measure(
                fn, q, k, v, device, args.warmup, args.iterations
            )
            outputs[name] = output
            records.append({
                "length": length,
                "method": name,
                "median_seconds": statistics.median(samples),
                "samples_seconds": samples,
                "peak_cuda_allocated_bytes": peak,
                "incremental_peak_cuda_bytes": None if peak is None else peak - baseline,
            })
        reference_output = outputs["reference"].float()
        for record in records[-len(methods):]:
            output = outputs[record["method"]].float()
            record["max_abs_error_vs_reference"] = float((output - reference_output).abs().max().item())
            record["mean_abs_error_vs_reference"] = float((output - reference_output).abs().mean().item())
            record["allclose_vs_reference"] = bool(torch.allclose(
                output, reference_output, rtol=1e-4 if dtype == torch.float32 else 2e-2,
                atol=1e-5 if dtype == torch.float32 else 2e-2
            ))
        # The educational kernels upcast half/bfloat16 arithmetic to float32
        # for stable softmax reductions, so score workspace uses compute dtype.
        score_element_size = 4 if dtype in (torch.float16, torch.bfloat16) else torch.empty((), dtype=dtype).element_size()
        score_bytes = args.batch * args.heads * length * length * score_element_size
        tile_score_bytes = args.batch * args.heads * min(length, args.tile_q) * min(length, args.tile_k) * score_element_size
        for record in records[-len(methods):]:
            record["materialized_score_tensor_bytes"] = score_bytes if record["method"] == "reference" else None
            record["maximum_tiled_score_tensor_bytes"] = tile_score_bytes if record["method"] == "tiled_online" else None

    payload = {
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "dtype": str(dtype),
        "shape": {"batch": args.batch, "heads": args.heads, "head_dim": args.head_dim},
        "causal": True,
        "tile_q": args.tile_q,
        "tile_k": args.tile_k,
        "warmup_iterations": args.warmup,
        "timed_iterations": args.iterations,
        "records": records,
    }
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for r in records:
        memory = (f"peak_cuda={r['peak_cuda_allocated_bytes']}" if r["peak_cuda_allocated_bytes"] is not None
                  else "peak_cuda=n/a (CPU)")
        print(f"len={r['length']:4d} {r['method']:12s} {r['median_seconds']:.6f}s "
              f"max_err={r['max_abs_error_vs_reference']:.3g} "
              f"allclose={r['allclose_vs_reference']} {memory}")
    print(f"Saved benchmark to {output_path}")


if __name__ == "__main__":
    main()
