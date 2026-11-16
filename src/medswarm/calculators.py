"""calculate(): run the official MedCalc-Bench implementation for a calculator on extracted params.

Params use the same keys/format as the dataset's "Relevant Entities"; a missing key means
"not mentioned", which the official code scores as absent (0 points).
"""
import importlib.util
import json
import sys
from functools import cache
from pathlib import Path

from medswarm.data import ROOT

IMPL_DIR = ROOT / "third_party" / "MedCalc-Bench" / "calculator_implementations"
sys.path.insert(0, str(IMPL_DIR))  # implementations import each other (e.g. age_conversion)


@cache
def _registry() -> dict:
    return json.loads((IMPL_DIR / "name_to_python.json").read_text(encoding="utf-8"))


@cache
def _function(calc_id: int):
    entry = _registry()[str(calc_id)]
    path = IMPL_DIR / Path(entry["file path"]).name
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, entry["explanation function"])


META_KEYS = {"file path", "explanation function", "calculator name", "type", "question"}


def variables(calc_id: int) -> dict:
    """Calculator spec: {entity name as in Relevant Entities -> python param name}."""
    return {k: v for k, v in _registry()[str(calc_id)].items() if k not in META_KEYS}


def calculate(calc_id: int, params: dict) -> float:
    """params keyed by entity names (as in Relevant Entities); missing keys are left out."""
    names = variables(calc_id)
    return _function(calc_id)({names[k]: v for k, v in params.items()})["Answer"]


def is_correct(answer, row) -> bool:
    """Official MedCalc-Bench rule: integers must match exactly, decimals must fall in [Lower, Upper]."""
    try:
        answer = float(answer)
    except (TypeError, ValueError):
        return False
    if row["Output Type"] == "integer":
        return round(answer) == round(float(row["Ground Truth Answer"]))
    return float(row["Lower Limit"]) <= answer <= float(row["Upper Limit"])
