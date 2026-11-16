"""Filter teacher episodes and turn them into chat samples for SFT.

Keep an episode only if: correct, label reproduced by the oracle, no runtime error, every sub-agent
ended with a valid extraction. At most one episode per row and CAP per calculator.
Each kept episode yields one orchestrator sample (sub-agent results appear as tool messages, which
assistant-only loss masks) and one sample per sub-agent (only its final valid answer).
The orchestrator's free text next to its tool calls is dropped by default: the teacher often reads the
note and extracts values itself before delegating the same work (83% of episodes have such text, 27%
quote the note), and the student should learn to delegate, not to imitate that.

Usage: python -m medswarm.build_sft runs/teacher_train*.jsonl --out data/sft
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from medswarm.prompts import TOOLS

CAP_PER_CALC = 200
VAL_FRACTION = 0.03


def keep(rec) -> bool:
    subs = [a for a in rec["agents"] if a["role"] == "subagent"]
    return (rec["correct"] and rec["oracle_consistent"] and not rec["excluded"] and rec["error"] is None
            and subs and all(not a["errors"] for a in subs))


def with_dict_args(messages):
    """Chat templates render tool-call arguments from a dict."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [{**c, "function": {**c["function"], "arguments": json.loads(c["function"]["arguments"])}}
                               for c in m["tool_calls"]]
        out.append(m)
    return out


def calls_only(messages):
    """Drop free text from tool-calling turns, and drop text-only turns (the teacher scoring by hand)
    together with the harness nudge that followed them, so results lead straight to calculate()."""
    out = []
    for i, m in enumerate(messages):
        if m["role"] == "assistant" and not m.get("tool_calls"):
            continue
        if m["role"] == "user" and i > 1:  # nudge after a text-only turn
            continue
        out.append({**m, "content": ""} if m["role"] == "assistant" else m)
    return out


def samples(rec, keep_text=False):
    for a in rec["agents"]:
        if a["role"] == "orchestrator":
            msgs = a["messages"] if keep_text else calls_only(a["messages"])
            yield {"messages": with_dict_args(msgs), "tools": TOOLS, "kind": "orchestrator"}
        else:  # system, user, final valid answer (drop failed attempts so they get no loss)
            yield {"messages": a["messages"][:2] + a["messages"][-1:], "kind": "subagent"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", default="data/sft")
    p.add_argument("--keep-orchestrator-text", action="store_true", help="ablation: keep the teacher's prose")
    a = p.parse_args()

    recs = [json.loads(l) for f in a.runs for l in open(f, encoding="utf-8")]
    kept, seen, per_calc = [], set(), Counter()
    random.seed(0)
    random.shuffle(recs)
    for r in recs:
        if keep(r) and r["row"] not in seen and per_calc[r["calc"]] < CAP_PER_CALC:
            kept.append(r)
            seen.add(r["row"])
            per_calc[r["calc"]] += 1
    print(f"kept {len(kept)}/{len(recs)} episodes")
    print("\n".join(f"  {n:4d}  {c}" for c, n in per_calc.most_common()))

    n_val = max(1, int(len(kept) * VAL_FRACTION))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("val", kept[:n_val]), ("train", kept[n_val:])):
        rows = [s for r in part for s in samples(r, a.keep_orchestrator_text)]
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(s, ensure_ascii=False) + "\n" for s in rows)
        print(f"{name}: {len(rows)} samples {dict(Counter(s['kind'] for s in rows))}")


if __name__ == "__main__":
    main()
