"""Render one episode as an agent tree (from a log, or live against a vLLM server).

Usage:
  python -m medswarm.demo runs/m_test.jsonl --row 12
  python -m medswarm.demo --live --row 12 --model runs/sft_qwen35_2b/merged
"""
import argparse
import asyncio
import json

from rich import print
from rich.tree import Tree

from medswarm.harness import LLM, run_episode, select_rows


def render(rec):
    ok = "[green]correct[/]" if rec["correct"] else "[red]wrong[/]"
    root = Tree(f"[bold]{rec['calc']}[/] | row {rec['row']} | {rec['mode']} | answer {rec['answer']} "
                f"(gold {rec['gold']}) {ok} | critical steps {rec['critical_steps']} | {rec['wall_s']:.1f}s")
    for agent in rec["agents"]:
        if agent["role"] != "subagent":
            continue
        node = root.add(f"[cyan]sub-agent {agent['id']}[/] -> {', '.join(agent['variables'])}")
        for name in agent["variables"]:
            value = agent.get("values", {}).get(name, "not mentioned")
            gold = rec["gold_entities"].get(name, "not mentioned")
            mark = "" if value == gold else f"  [yellow](gold: {gold})[/]"
            node.add(f"{name} = {value}{mark}")
        for err in agent.get("errors", []):
            node.add(f"[red]{err}[/]")
    print(root)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log", nargs="?")
    p.add_argument("--row", type=int, required=True)
    p.add_argument("--live", action="store_true")
    p.add_argument("--model")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    a = p.parse_args()
    if a.live:
        df = select_rows("test")
        row = df[df["Row Number"] == a.row].iloc[0]
        rec = asyncio.run(run_episode(LLM(a.base_url, a.model), "orchestrator", row))
    else:
        rec = next(r for r in map(json.loads, open(a.log, encoding="utf-8")) if r["row"] == a.row)
    render(rec)


if __name__ == "__main__":
    main()
