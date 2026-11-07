"""Criterion-level eligibility and evidence evaluation.

The model predicts the dataset's expert category and sentence IDs. Evidence
text and audit explanations are then attached deterministically from the
patient note so the explanation cannot introduce facts beyond that note.
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from common import prompt_for, parse_evidence


UNRESOLVED_LABELS = {"not enough information", "not applicable"}
OPPOSING_LABELS = {
    "inclusion": {"included", "not included"},
    "exclusion": {"excluded", "not excluded"},
}


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def output_fields(text):
    text = text.strip()
    try:
        start, end = text.find("{"), text.rfind("}") + 1
        obj = json.loads(text[start:end]) if start >= 0 and end > start else {}
    except (json.JSONDecodeError, TypeError, ValueError):
        obj = {}
    label = str(obj.get("eligibility", "invalid")).strip().lower()
    evidence = parse_evidence(obj.get("evidence_sentence_ids", []))
    return label, evidence


def sentence_map(note):
    """Read the benchmark's numbered note sentences without paraphrasing."""
    note = str(note or "")
    matches = list(re.finditer(r"(?<![\w.])(\d+)\.\s+", note))
    result = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(note)
        sentence_id = int(match.group(1))
        sentence = note[match.end():end].strip()
        if sentence:
            result.setdefault(sentence_id, sentence)
    return result


def explanation_for(label, criterion_type, criterion, cited, invalid_ids):
    description = f"The model predicted '{label}' for this {criterion_type} criterion: {criterion}."
    if invalid_ids:
        invalid_note = f" It also returned sentence ID(s) {', '.join(map(str, invalid_ids))}, which do not match a numbered sentence in the supplied note."
    else:
        invalid_note = ""
    if cited:
        ids = ", ".join(str(item["sentence_id"]) for item in cited)
        quotes = " ".join(f"[{item['sentence_id']}] {item['text']}" for item in cited)
        return f"{description} Its cited patient-note sentence IDs are {ids}: {quotes}{invalid_note}"
    if label == "not applicable":
        return f"{description} The model cited no verifiable patient sentence.{invalid_note} 'Not applicable' is retained as a separate category; missing evidence does not establish either eligibility or ineligibility."
    if label == "not enough information":
        return f"{description} The model cited no verifiable patient sentence.{invalid_note} Missing evidence is not treated as proof of eligibility or ineligibility."
    return f"{description} The model cited no verifiable patient sentence.{invalid_note} That absence is not treated as proof of eligibility or ineligibility."


def classification_metrics(examples):
    if not examples:
        return {"n": 0, "accuracy": None, "macro_f1": None, "labels": [], "per_class": {}, "confusion_matrix": []}
    gold = [item["gold_label"] for item in examples]
    predicted = [item["predicted_category"] for item in examples]
    labels = sorted(set(gold) | set(predicted))
    return {
        "n": len(examples),
        "accuracy": float(accuracy_score(gold, predicted)),
        "macro_f1": float(f1_score(gold, predicted, labels=labels, average="macro", zero_division=0)),
        "labels": labels,
        "per_class": classification_report(gold, predicted, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(gold, predicted, labels=labels).tolist(),
    }


def evidence_metrics(examples):
    true_positive = sum(len(set(x["gold_evidence_ids"]) & set(x["evidence_sentence_ids"])) for x in examples)
    predicted_total = sum(len(x["evidence_sentence_ids"]) for x in examples)
    gold_total = sum(len(x["gold_evidence_ids"]) for x in examples)
    # Micro set metrics: if both sides have no evidence, all three are 1;
    # if only one side is empty, the corresponding precision/recall is 0.
    precision = true_positive / predicted_total if predicted_total else (1.0 if gold_total == 0 else 0.0)
    recall = true_positive / gold_total if gold_total else (1.0 if predicted_total == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    both_empty = sum(not x["gold_evidence_ids"] and not x["evidence_sentence_ids"] for x in examples)
    gold_empty_predicted = sum(not x["gold_evidence_ids"] and bool(x["evidence_sentence_ids"]) for x in examples)
    gold_present_predicted_empty = sum(bool(x["gold_evidence_ids"]) and not x["evidence_sentence_ids"] for x in examples)
    return {
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "gold_evidence_empty_records": sum(not x["gold_evidence_ids"] for x in examples),
        "predicted_evidence_empty_records": sum(not x["evidence_sentence_ids"] for x in examples),
        "both_empty_records": both_empty,
        "gold_empty_predicted_nonempty_records": gold_empty_predicted,
        "gold_nonempty_predicted_empty_records": gold_present_predicted_empty,
        "empty_evidence_policy": "Micro exact-ID set counts: both-empty corpus gives precision/recall/F1=1; one-sided empty gives the affected metric 0.",
    }


def error_kind(item):
    gold, pred = item["gold_label"], item["predicted_category"]
    criterion_type = item["criterion_type"].lower()
    if gold == pred:
        return None
    opposing = OPPOSING_LABELS.get(criterion_type, set())
    if gold in opposing and pred in opposing:
        return "opposing_definitive_categories"
    if gold == "not enough information" and pred in opposing:
        return "missing_information_overclaim"
    return "other_category_mismatch"


def error_analysis(examples):
    errors = []
    for item in examples:
        kind = error_kind(item)
        if kind:
            errors.append({"error_kind": kind, **item})
    summary = {}
    for kind in ("opposing_definitive_categories", "missing_information_overclaim", "other_category_mismatch"):
        cases = [item for item in errors if item["error_kind"] == kind]
        summary[kind] = {"count": len(cases), "representative_cases": cases[:5]}
    missing = [item for item in examples if item["gold_label"] == "not enough information"]
    overclaims = [item for item in errors if item["error_kind"] == "missing_information_overclaim"]
    summary["gold_not_enough_information"] = {
        "count": len(missing),
        "correctly_preserved": sum(item["predicted_category"] == "not enough information" for item in missing),
        "predicted_as_definitive": len(overclaims),
        "definitive_predictions_with_evidence": sum(bool(item["evidence_sentence_ids"]) for item in overclaims),
        "definitive_predictions_without_evidence": sum(not item["evidence_sentence_ids"] for item in overclaims),
        "representative_failures": overclaims[:5],
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen_lora.json")
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--mode", choices=["base", "finetuned"], default="finetuned")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    rows = read_rows(f"data/processed/{args.split}.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForCausalLM.from_pretrained(config["model_name"], torch_dtype="auto", device_map="auto")
    if args.mode == "finetuned":
        model = PeftModel.from_pretrained(model, f"{config['output_dir']}/best_adapter")
    model.eval()

    examples, truncations, latencies = [], [], []
    for row in rows:
        prompt = prompt_for(row)
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=config["max_length"] - config["max_new_tokens"])
        truncations.append(len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) > config["max_length"] - config["max_new_tokens"])
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**encoded, max_new_tokens=config["max_new_tokens"], do_sample=False, pad_token_id=tokenizer.eos_token_id, use_cache=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - start)
        raw = tokenizer.decode(generated[0, encoded["input_ids"].shape[-1]:], skip_special_tokens=True)
        predicted_category, evidence_ids = output_fields(raw)
        sentences = sentence_map(row["note"])
        cited = [{"sentence_id": sentence_id, "text": sentences[sentence_id]} for sentence_id in evidence_ids if sentence_id in sentences]
        invalid_evidence_ids = [sentence_id for sentence_id in evidence_ids if sentence_id not in sentences]
        gold_ids = parse_evidence(row["expert_sentences"])
        criterion_type = str(row["criterion_type"]).strip().lower()
        example = {
            "patient_id": row["patient_id"],
            "criterion_type": criterion_type,
            "criterion": row["criterion_text"],
            "gold_label": str(row["expert_eligibility"]).strip().lower(),
            "predicted_category": predicted_category,
            "cited_sentence_ids": evidence_ids,
            "evidence_sentence_ids": evidence_ids,
            "evidence_text": cited,
            "invalid_evidence_sentence_ids": invalid_evidence_ids,
            "category_valid_for_criterion_type": predicted_category in (
                {"included", "not included", "not enough information", "not applicable"}
                if criterion_type == "inclusion"
                else {"excluded", "not excluded", "not enough information", "not applicable"}
                if criterion_type == "exclusion"
                else set()
            ),
            "explanation": explanation_for(predicted_category, criterion_type, row["criterion_text"], cited, invalid_evidence_ids),
            "gold_evidence_ids": gold_ids,
            "raw_output": raw,
            "truncated": truncations[-1],
        }
        examples.append(example)

    by_type = {}
    for criterion_type in ("inclusion", "exclusion"):
        group = [item for item in examples if item["criterion_type"] == criterion_type]
        by_type[criterion_type] = {"classification": classification_metrics(group), "evidence": evidence_metrics(group)}
    overall = {
        "classification": classification_metrics(examples),
        "evidence": evidence_metrics(examples),
    }
    metrics = {
        "split": args.split,
        "mode": args.mode,
        "n": len(examples),
        "overall": overall,
        "by_criterion_type": by_type,
        "truncated_fraction": float(np.mean(truncations)) if truncations else 0.0,
        "mean_generation_seconds": float(np.mean(latencies)) if latencies else 0.0,
        "error_analysis": error_analysis(examples),
        "predictions": examples,
    }
    output_dir = Path("outputs/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.split}_{args.mode}.json"
    output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    concise = {key: value for key, value in metrics.items() if key not in ("predictions", "error_analysis")}
    print(json.dumps(concise, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
