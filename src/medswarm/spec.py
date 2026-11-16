"""Per-calculator variable spec (derived from train Relevant Entities) and the extraction validator.

An extraction is {entity name: {"value": ..., "evidence": "<verbatim quote from the note>"}}.
value=None means "not mentioned" (the official code then scores it as absent).
"""
import json
import re
from collections import defaultdict
from functools import cache

from medswarm.calculators import variables
from medswarm.data import ROOT, load

SPEC_PATH = ROOT / "data" / "spec.json"
# Fixes where train data under-covers what the official code accepts (from heart_score.py) or is noisy.
OVERRIDES = {
    ("18", "Electrocardiogram Test"): {"type": "choice", "options": [
        "Normal", "Non-specific repolarization disturbance", "Significant ST deviation"]},
    ("51", "respiratory rate"): {"type": "number_with_unit", "units": [
        "breaths per minute", "breaths/min", "breaths/minute", "cycles per minute"]},
}


def build_spec() -> dict:
    seen = defaultdict(list)
    df = load("train")
    for cid, ents in zip(df["Calculator ID"], df["entities"]):
        for name, v in ents.items():
            seen[(cid, name)].append(v)
    spec = {}
    for cid in sorted(df["Calculator ID"].unique()):
        spec[str(cid)] = {}
        for name in variables(cid):
            vals = seen.get((cid, name), [])
            if vals and all(isinstance(v, bool) for v in vals):
                s = {"type": "boolean"}
            elif vals and all(isinstance(v, str) for v in vals):
                s = {"type": "choice", "options": sorted(set(vals))}
            elif vals and all(isinstance(v, list) for v in vals):
                s = {"type": "number_with_unit", "units": sorted({v[1] for v in vals})}
            elif vals and all(isinstance(v, (int, float)) for v in vals):
                s = {"type": "number"}
            else:  # never seen in train: accept a number in any unit
                s = {"type": "number_with_unit", "units": []}
            spec[str(cid)][name] = OVERRIDES.get((str(cid), name), s)
    return spec


@cache
def get_spec() -> dict:
    if not SPEC_PATH.exists():
        SPEC_PATH.write_text(json.dumps(build_spec(), indent=1, ensure_ascii=False), encoding="utf-8")
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def validate(extraction, calc_id: int, note: str, expected: list[str]) -> tuple[dict, list[str]]:
    """Return (clean values keyed by entity name, without not-mentioned ones; list of error strings)."""
    if not isinstance(extraction, dict):
        return {}, ["Output must be a JSON object keyed by variable name."]
    spec, values, errors = get_spec()[str(calc_id)], {}, []
    for name in expected:
        if name not in extraction:
            errors.append(f"Missing variable '{name}' (use value null if not mentioned).")
            continue
        item = extraction[name]
        if not isinstance(item, dict) or "value" not in item:
            errors.append(f"'{name}' must be an object with 'value' and 'evidence'.")
            continue
        v, s = item["value"], spec[name]
        if v is None:
            continue
        ok = {
            "boolean": isinstance(v, bool),
            "choice": v in s.get("options", []),
            "number": isinstance(v, (int, float)) and not isinstance(v, bool),
            "number_with_unit": isinstance(v, list) and len(v) == 2
            and isinstance(v[0], (int, float)) and not isinstance(v[0], bool)
            and (not s.get("units") or v[1] in s["units"]),
        }[s["type"]]
        if not ok:
            errors.append(f"Invalid value for '{name}': {json.dumps(v)}. Expected {describe(s)}.")
            continue
        evidence = item.get("evidence") or ""
        if not (isinstance(evidence, str) and evidence.strip() and _norm(evidence) in _norm(note)):
            errors.append(f"Evidence for '{name}' must be a verbatim quote from the note.")
            continue
        values[name] = v
    return values, errors


def describe(s: dict) -> str:
    if s["type"] == "boolean":
        return "true or false"
    if s["type"] == "choice":
        return "one of " + json.dumps(s["options"])
    if s["type"] == "number":
        return "a number"
    units = json.dumps(s["units"]) if s["units"] else "any unit"
    return f"[number, unit] with unit in {units}"
