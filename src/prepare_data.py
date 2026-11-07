import argparse, json
from pathlib import Path
import numpy as np
from datasets import load_dataset

REQUIRED = {"patient_id", "note", "criterion_type", "criterion_text", "expert_eligibility", "expert_sentences"}
FORBIDDEN = {"gpt4_explanation", "gpt4_sentences", "gpt4_eligibility", "explanation_correctness"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("ncbi/TrialGPT-Criterion-Annotations", split="train")
    cols = set(ds.column_names)
    if missing := REQUIRED - cols: raise ValueError(f"Dataset missing columns: {sorted(missing)}")
    patients = sorted(set(ds["patient_id"]))
    rng = np.random.default_rng(42); shuffled = list(np.asarray(patients)[rng.permutation(len(patients))])
    train, val, test = shuffled[:37], shuffled[37:45], shuffled[45:53]
    if len(patients) != 53 or len(train) != 37 or len(val) != 8 or len(test) != 8:
        raise ValueError(f"Expected 53 patients and 37/8/8 split; found {len(patients)}")
    assign = {p: s for s, ps in [("train", train), ("validation", val), ("test", test)] for p in ps}
    manifest = {"seed":42,"method":"sorted unique IDs, then numpy.default_rng(42).permutation","patient_counts":{"train":len(train),"validation":len(val),"test":len(test)},"row_counts":{},"patients":{"train":train,"validation":val,"test":test}}
    splits = {k: [] for k in ("train", "validation", "test")}
    # Deliberately whitelist usable fields; no model-generated source fields survive.
    for row in ds:
        clean = {k: row[k] for k in REQUIRED}
        clean["split"] = assign[row["patient_id"]]
        splits[clean["split"]].append(clean)
    for name, rows in splits.items():
        manifest["row_counts"][name] = len(rows)
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows: f.write(json.dumps(row, ensure_ascii=False) + "\n")
    groups = [set(train), set(val), set(test)]
    assert not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
    assert all(not (FORBIDDEN & set(r)) for rows in splits.values() for r in rows)
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
if __name__ == "__main__": main()
