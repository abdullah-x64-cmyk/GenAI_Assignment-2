"""Create a submission archive without unstructured generations that may quote notes."""
import argparse
import json
import zipfile
from pathlib import Path

from analyze_saved_evaluations import convert_rows, summarize


ALLOWED = {
    "inclusion": {"included", "not included", "not enough information", "not applicable"},
    "exclusion": {"excluded", "not excluded", "not enough information", "not applicable"},
}


def scrub_evaluation(entry_name, payload):
    document = json.loads(payload)
    mode = document.get("mode", "unknown")
    if mode == "base":
        rows = document.pop("predictions", [])
        counts = {"inclusion": 0, "exclusion": 0, "other": 0}
        for row in rows:
            criterion_type = str(row.get("criterion_type", "other")).strip().lower()
            group = criterion_type if criterion_type in ("inclusion", "exclusion") else "other"
            label = str(row.get("predicted_label", row.get("predicted_category", "invalid"))).strip().lower()
            if label not in ALLOWED.get(criterion_type, set()):
                counts[group] += 1
        # Preserve the aggregate benchmark result and redact labels, class
        # reports, and raw generations whose arbitrary text can quote a note.
        keys = (
            "split", "mode", "n", "accuracy", "macro_f1",
            "evidence_micro_precision", "evidence_micro_recall", "evidence_micro_f1",
            "truncated_fraction", "mean_generation_seconds",
        )
        safe = {key: document[key] for key in keys if key in document}
        safe["unstructured_or_type_incompatible_generation_counts"] = counts
        safe["redaction"] = "Per-case baseline generations and class labels omitted because malformed free-form outputs may reproduce patient-note text. Aggregate metrics are retained unchanged."
        safe_input = dict(document)
        safe_input["predictions"] = rows
        safe["safe_reaggregation"] = summarize(convert_rows(safe_input))
        return json.dumps(safe, indent=2, ensure_ascii=False).encode("utf-8")

    # Fine-tuned predictions followed the expected short JSON shape in this
    # run. Remove raw generations from the public copy regardless.
    for row in document.get("predictions", []):
        row.pop("raw_output", None)
    return json.dumps(document, indent=2, ensure_ascii=False).encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source", help="Existing artifact archive")
    source_group.add_argument("--root", help="Project folder containing outputs/ and data/processed/split_manifest.json")
    parser.add_argument("--out", required=True, help="Sanitized output archive")
    args = parser.parse_args()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as target_zip:
        if args.source:
            source = Path(args.source)
            with zipfile.ZipFile(source, "r") as source_zip:
                for item in source_zip.infolist():
                    payload = source_zip.read(item.filename)
                    if item.filename.startswith("outputs/evaluation/") and item.filename.endswith(".json"):
                        payload = scrub_evaluation(item.filename, payload)
                    target_zip.writestr(item, payload)
        else:
            root = Path(args.root)
            for path in (root / "outputs").rglob("*"):
                if path.is_file():
                    name = path.relative_to(root).as_posix()
                    payload = path.read_bytes()
                    if name.startswith("outputs/evaluation/") and name.endswith(".json"):
                        payload = scrub_evaluation(name, payload)
                    target_zip.writestr(name, payload)
            manifest = root / "data/processed/split_manifest.json"
            if manifest.exists():
                target_zip.write(manifest, manifest.relative_to(root).as_posix())
    print(f"Created sanitized artifact bundle: {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
