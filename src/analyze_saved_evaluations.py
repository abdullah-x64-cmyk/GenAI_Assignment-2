"""Re-aggregate existing prediction logs without rerunning inference."""
import argparse
import json
import zipfile
import warnings
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

warnings.filterwarnings("ignore", message="The number of unique classes is greater than 50%")


UNRESOLVED = {"not enough information", "not applicable"}
OPPOSING = {"inclusion": {"included", "not included"}, "exclusion": {"excluded", "not excluded"}}


def classification(rows):
    if not rows:
        return {"n": 0, "accuracy": None, "macro_f1": None, "labels": [], "per_class": {}, "confusion_matrix": []}
    gold = [row["gold_label"] for row in rows]
    pred = [row["predicted_category"] for row in rows]
    labels = sorted(set(gold) | set(pred))
    return {
        "n": len(rows),
        "accuracy": float(accuracy_score(gold, pred)),
        "macro_f1": float(f1_score(gold, pred, labels=labels, average="macro", zero_division=0)),
        "labels": labels,
        "per_class": classification_report(gold, pred, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(gold, pred, labels=labels).tolist(),
    }


def evidence(rows):
    tp = sum(len(set(row["gold_evidence_ids"]) & set(row["evidence_sentence_ids"])) for row in rows)
    pred_n = sum(len(row["evidence_sentence_ids"]) for row in rows)
    gold_n = sum(len(row["gold_evidence_ids"]) for row in rows)
    p = tp / pred_n if pred_n else (1.0 if gold_n == 0 else 0.0)
    r = tp / gold_n if gold_n else (1.0 if pred_n == 0 else 0.0)
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {
        "precision": p, "recall": r, "f1": f,
        "gold_evidence_empty_records": sum(not row["gold_evidence_ids"] for row in rows),
        "predicted_evidence_empty_records": sum(not row["evidence_sentence_ids"] for row in rows),
        "both_empty_records": sum(not row["gold_evidence_ids"] and not row["evidence_sentence_ids"] for row in rows),
        "gold_empty_predicted_nonempty_records": sum(not row["gold_evidence_ids"] and bool(row["evidence_sentence_ids"]) for row in rows),
        "gold_nonempty_predicted_empty_records": sum(bool(row["gold_evidence_ids"]) and not row["evidence_sentence_ids"] for row in rows),
        "empty_evidence_policy": "Micro exact-ID set counts: both-empty corpus gives precision/recall/F1=1; one-sided empty gives the affected metric 0.",
    }


def kind(row):
    gold, pred = row["gold_label"], row["predicted_category"]
    opposing = OPPOSING.get(row["criterion_type"], set())
    if gold == pred:
        return None
    if gold in opposing and pred in opposing:
        return "opposing_definitive_categories"
    if gold == "not enough information" and pred in opposing:
        return "missing_information_overclaim"
    return "other_category_mismatch"


def read_json_from_archive(path, entry):
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read(entry))


def convert_rows(document):
    converted = []
    for item in document["predictions"]:
        criterion_type = str(item.get("criterion_type", "unknown")).strip().lower()
        predicted = str(item.get("predicted_category", item.get("predicted_label", "invalid"))).strip().lower()
        allowed = {
            "inclusion": {"included", "not included", "not enough information", "not applicable"},
            "exclusion": {"excluded", "not excluded", "not enough information", "not applicable"},
        }.get(criterion_type, set())
        # Malformed baseline generations may contain copied note sentences.
        # Collapse these only for the public audit artifact, not source metrics.
        if predicted not in allowed:
            predicted = "invalid or unstructured"
        converted.append({
            "patient_id": item.get("patient_id"),
            "criterion_type": criterion_type,
            "criterion": item.get("criterion", item.get("criterion_text")),
            "gold_label": str(item["gold_label"]).strip().lower(),
            "predicted_category": predicted,
            "gold_evidence_ids": [int(x) for x in item.get("gold_evidence_ids", [])],
            "evidence_sentence_ids": [int(x) for x in item.get("evidence_sentence_ids", item.get("predicted_evidence_ids", []))],
            "evidence_text": item.get("evidence_text", []),
        })
    return converted


def summarize(rows):
    by_type = {}
    for criterion_type in ("inclusion", "exclusion"):
        group = [row for row in rows if row["criterion_type"] == criterion_type]
        by_type[criterion_type] = {"classification": classification(group), "evidence": evidence(group)}
    errors = []
    for row in rows:
        error_kind = kind(row)
        if error_kind:
            errors.append({"error_kind": error_kind, **row})
    analysis = {}
    for error_kind in ("opposing_definitive_categories", "missing_information_overclaim", "other_category_mismatch"):
        cases = [row for row in errors if row["error_kind"] == error_kind]
        analysis[error_kind] = {"count": len(cases), "representative_cases": cases[:5]}
    missing = [row for row in rows if row["gold_label"] == "not enough information"]
    overclaims = [row for row in errors if row["error_kind"] == "missing_information_overclaim"]
    analysis["gold_not_enough_information"] = {
        "count": len(missing),
        "correctly_preserved": sum(row["predicted_category"] == "not enough information" for row in missing),
        "predicted_as_definitive": len(overclaims),
        "definitive_predictions_with_evidence": sum(bool(row["evidence_sentence_ids"]) for row in overclaims),
        "definitive_predictions_without_evidence": sum(not row["evidence_sentence_ids"] for row in overclaims),
        "representative_failures": overclaims[:5],
    }
    return {"overall": {"classification": classification(rows), "evidence": evidence(rows)}, "by_criterion_type": by_type, "error_analysis": analysis}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="results/task1_submission_artifacts.zip")
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument("--out", default="results/task2_error_analysis.json")
    args = parser.parse_args()
    archive = Path(args.archive)
    results = {"source_archive": archive.name, "split": args.split, "models": {}}
    for mode in ("base", "finetuned"):
        entry = f"outputs/evaluation/{args.split}_{mode}.json"
        document = read_json_from_archive(archive, entry)
        if "predictions" in document:
            rows = convert_rows(document)
            results["models"][mode] = summarize(rows)
        elif "safe_reaggregation" in document:
            results["models"][mode] = document["safe_reaggregation"]
        else:
            raise ValueError(f"{entry} contains neither predictions nor a safe aggregate")
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    compact = {}
    for mode, result in results["models"].items():
        overall = result["overall"]
        compact[mode] = {
            "n": overall["classification"]["n"],
            "accuracy": overall["classification"]["accuracy"],
            "macro_f1": overall["classification"]["macro_f1"],
            "evidence": overall["evidence"],
            "by_criterion_type": {
                key: {"n": value["classification"]["n"], "accuracy": value["classification"]["accuracy"], "macro_f1": value["classification"]["macro_f1"], "evidence_f1": value["evidence"]["f1"]}
                for key, value in result["by_criterion_type"].items()
            },
            "error_counts": {key: value["count"] for key, value in result["error_analysis"].items() if "count" in value},
        }
    print(json.dumps(compact, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
