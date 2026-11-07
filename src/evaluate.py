import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from common import prompt_for, parse_evidence

def read_rows(path): return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x]

def output_fields(text):
    text = text.strip()
    try:
        obj = json.loads(text[text.find("{"):text.rfind("}") + 1])
    except Exception:
        obj = {}
    label = str(obj.get("eligibility", "invalid")).strip().lower()
    evidence = parse_evidence(obj.get("evidence_sentence_ids", []))
    return label, evidence

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen_lora.json")
    ap.add_argument("--split", choices=["validation", "test"], required=True)
    ap.add_argument("--mode", choices=["base", "finetuned"], default="finetuned")
    args = ap.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    rows = read_rows(f"data/processed/{args.split}.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForCausalLM.from_pretrained(config["model_name"], torch_dtype="auto", device_map="auto")
    if args.mode == "finetuned":
        model = PeftModel.from_pretrained(model, f"{config['output_dir']}/best_adapter")
    model.eval()

    gold, predicted, gold_evidence, predicted_evidence = [], [], [], []
    outputs, truncations, latencies = [], [], []
    for row in rows:
        prompt = prompt_for(row)
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=config["max_length"] - config["max_new_tokens"])
        truncated = len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) > config["max_length"] - config["max_new_tokens"]
        truncations.append(truncated)
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        if torch.cuda.is_available(): torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**encoded, max_new_tokens=config["max_new_tokens"], do_sample=False, pad_token_id=tokenizer.eos_token_id, use_cache=True)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        latencies.append(time.perf_counter() - start)
        raw = tokenizer.decode(generated[0, encoded["input_ids"].shape[-1]:], skip_special_tokens=True)
        label, evidence_ids = output_fields(raw)
        gold_label = str(row["expert_eligibility"]).strip().lower()
        gold_ids = parse_evidence(row["expert_sentences"])
        gold.append(gold_label); predicted.append(label)
        gold_evidence.append(set(gold_ids)); predicted_evidence.append(set(evidence_ids))
        outputs.append({
            "patient_id": row["patient_id"], "criterion_type": row["criterion_type"],
            "gold_label": gold_label, "predicted_label": label,
            "gold_evidence_ids": sorted(gold_ids), "predicted_evidence_ids": sorted(evidence_ids),
            "raw_output": raw, "truncated": truncated,
        })

    labels = sorted(set(gold) | set(predicted))
    true_positive = sum(len(x & y) for x, y in zip(gold_evidence, predicted_evidence))
    predicted_total = sum(map(len, predicted_evidence)); gold_total = sum(map(len, gold_evidence))
    precision = true_positive / predicted_total if predicted_total else (1.0 if gold_total == 0 else 0.0)
    recall = true_positive / gold_total if gold_total else (1.0 if predicted_total == 0 else 0.0)
    evidence_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {
        "split": args.split, "mode": args.mode, "n": len(rows),
        "accuracy": accuracy_score(gold, predicted),
        "macro_f1": f1_score(gold, predicted, labels=labels, average="macro", zero_division=0),
        "class_report": classification_report(gold, predicted, labels=labels, output_dict=True, zero_division=0),
        "labels": labels, "confusion_matrix": confusion_matrix(gold, predicted, labels=labels).tolist(),
        "evidence_micro_precision": precision, "evidence_micro_recall": recall, "evidence_micro_f1": evidence_f1,
        "truncated_fraction": float(np.mean(truncations)) if truncations else 0.0,
        "mean_generation_seconds": float(np.mean(latencies)) if latencies else 0.0,
        "predictions": outputs,
    }
    output_dir = Path("outputs/evaluation"); output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.split}_{args.mode}.json"
    output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    concise = {key: value for key, value in metrics.items() if key not in ("class_report", "confusion_matrix", "predictions")}
    print(json.dumps(concise, indent=2, ensure_ascii=False))

if __name__ == "__main__": main()
