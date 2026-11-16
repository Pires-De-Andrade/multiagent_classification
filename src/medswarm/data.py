"""Load MedCalc-Bench Verified (pinned HF commit) and keep the rule-based calculators."""
import ast
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
HF_COMMIT = "58fc3de568bfc53a68f32327e58f60dc6eaa17f8"  # nsk7153/MedCalc-Bench-Verified
RULE_BASED = {"risk", "severity", "diagnosis"}
EXCLUDED = {"APACHE II Score", "Caprini Score for Venous Thromboembolism (2005)"}  # corrupted labels, reported separately


def load(split: str, rule_based_only: bool = True) -> pd.DataFrame:
    df = pd.read_parquet(RAW / f"{split}.parquet")
    df["entities"] = df["Relevant Entities"].map(ast.literal_eval)
    if rule_based_only:
        df = df[df["Category"].isin(RULE_BASED)]
    return df.reset_index(drop=True)
