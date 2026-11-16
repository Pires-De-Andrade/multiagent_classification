"""Compare conditions (one JSONL of episodes per condition) on the rule-based test set.

Usage: python -m medswarm.evaluate B0=runs/b0.jsonl B1=runs/b1.jsonl M=runs/m.jsonl --baseline B1
"""
import argparse
import json
from math import comb

import numpy as np
import pandas as pd

from medswarm.spec import get_spec


def same_value(pred, gold, kind) -> bool:
    if kind == "boolean":
        return bool(pred) == bool(gold)  # missing == absent, as the official code scores it
    if pred is None or gold is None:
        return pred is None and gold is None
    if kind == "choice":
        return str(pred).lower() == str(gold).lower()
    if kind == "number_with_unit":
        return pred[1] == gold[1] and abs(pred[0] - gold[0]) <= 0.05 * max(abs(gold[0]), 1e-9)
    return abs(float(pred) - float(gold)) <= 0.05 * max(abs(float(gold)), 1e-9)


def variable_accuracy(rec):
    """Accuracy over the variables the gold annotation mentions (None for CoT, which extracts nothing)."""
    if rec["mode"] == "cot" or rec["error"]:
        return None
    spec = get_spec()[str(rec["calc_id"])]
    gold = rec["gold_entities"]
    hits = [same_value(rec["values"].get(n), gold[n], spec[n]["type"]) for n in gold if n in spec]
    return float(np.mean(hits)) if hits else None


def load(path) -> pd.DataFrame:
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    for r in recs:
        r["var_acc"] = variable_accuracy(r)
        del r["agents"]
    return pd.DataFrame(recs).set_index("row").sort_index()


def mcnemar_exact(a: pd.Series, b: pd.Series) -> float:
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    n, k = n01 + n10, min(n01, n10)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def bootstrap_diff(a, b, iters=10_000, seed=0):
    rng = np.random.default_rng(seed)
    d = a.astype(float).values - b.astype(float).values
    means = d[rng.integers(0, len(d), (iters, len(d)))].mean(axis=1)
    return d.mean(), *np.percentile(means, [2.5, 97.5])


def holm(pvals: dict) -> dict:
    out, running = {}, 0.0
    for i, (k, p) in enumerate(sorted(pvals.items(), key=lambda kv: kv[1])):
        running = max(running, min(1.0, (len(pvals) - i) * p))
        out[k] = running
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="NAME=path.jsonl")
    p.add_argument("--baseline", required=True)
    a = p.parse_args()
    runs = {name: load(path) for name, path in (r.split("=", 1) for r in a.runs)}

    rows = []
    for name, df in runs.items():
        main_set = df[~df["excluded"]]
        rows.append({
            "condition": name, "n": len(main_set),
            "acc": main_set["correct"].mean(),
            "acc_oracle_consistent": main_set[main_set["oracle_consistent"]]["correct"].mean(),
            "acc_excluded_calcs": df[df["excluded"]]["correct"].mean(),
            "var_acc": main_set["var_acc"].mean(),
            "subagents": main_set["n_subagents"].mean(),
            "critical_steps": main_set["critical_steps"].mean(),
            "critical_tokens": main_set["critical_tokens"].mean(),
            "wall_s": main_set["wall_s"].mean(),
            "no_answer": main_set["answer"].isna().mean(),
            "errors": main_set["error"].notna().mean(),
        })
    print(pd.DataFrame(rows).set_index("condition").round(3).to_string())

    for name, df in runs.items():
        main_set = df[~df["excluded"]]
        print(f"\n{name} by note type / calculator:")
        print(main_set.groupby("note_type")["correct"].mean().round(3).to_dict())
        print(main_set.groupby("calc")["correct"].mean().round(2).sort_values().to_string())

    base = runs[a.baseline]
    pvals, diffs = {}, {}
    for name, df in runs.items():
        if name == a.baseline:
            continue
        common = df.index.intersection(base.index)
        common = common[~df.loc[common, "excluded"]]
        x, y = df.loc[common, "correct"].astype(bool), base.loc[common, "correct"].astype(bool)
        pvals[name], diffs[name] = mcnemar_exact(x, y), bootstrap_diff(x, y)
    adjusted = holm(pvals)
    print(f"\nPaired vs {a.baseline} (exact McNemar, Holm-adjusted; bootstrap 95% CI of accuracy difference):")
    for name in pvals:
        d, lo, hi = diffs[name]
        print(f"  {name}: diff={d:+.3f} [{lo:+.3f}, {hi:+.3f}]  p={pvals[name]:.4f}  p_holm={adjusted[name]:.4f}")


if __name__ == "__main__":
    main()
