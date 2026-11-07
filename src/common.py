import json, re

LABELS = ["included", "excluded", "not enough information", "not applicable"]

def parse_evidence(value):
    # The dataset can contain non-sentence identifiers (for example dotted
    # IDs) in evidence fields. Keep only integer sentence indices instead of
    # crashing or turning a dotted ID into several misleading sentence IDs.
    def valid_ids(items):
        result = []
        for item in items:
            text = str(item).strip()
            if re.fullmatch(r"\d+", text):
                result.append(int(text))
        return result
    if isinstance(value, list): return valid_ids(value)
    if value is None: return []
    if isinstance(value, (int, float)): return [int(value)]
    try:
        obj = json.loads(value)
        if isinstance(obj, list): return valid_ids(obj)
    except Exception: pass
    return [int(x) for x in re.findall(r"(?<![\w.])\d+(?![\w.])", str(value))]

def target_for(row):
    label = str(row["expert_eligibility"]).strip().lower()
    return json.dumps({"eligibility": label, "evidence_sentence_ids": parse_evidence(row["expert_sentences"])}, ensure_ascii=False)

def prompt_for(row):
    return ("You are a clinical-trial criterion screening research model. Use only the patient note. "
            "Return one JSON object with eligibility and evidence_sentence_ids. Eligibility must preserve the expert label category; "
            "do not infer from missing information. Evidence IDs must be sentence numbers explicitly present in the note.\n"
            f"Patient note:\n{row['note']}\nCriterion type: {row['criterion_type']}\nCriterion: {row['criterion_text']}\nJSON response:")
