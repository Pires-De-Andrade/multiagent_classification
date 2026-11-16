"""Async agent harness: orchestrator + parallel sub-agents, single-agent and CoT baselines.

Each episode is one JSON record with every agent's messages (linked by parent id), so the same
logs serve for SFT data, evaluation and the demo.

Usage:
  python -m medswarm.harness --mode orchestrator --split train --model Qwen/... --out runs/teacher_train.jsonl
"""
import argparse
import asyncio
import io
import json
import re
import time
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from medswarm import prompts
from medswarm.calculators import calculate, is_correct
from medswarm.data import EXCLUDED, ROOT, load
from medswarm.spec import get_spec, validate

MAX_ORCH_TURNS = 4
MAX_SUB_ATTEMPTS = 2
VOTES = 1               # --votes: independent extractions per sub-agent call, majority per variable
VOTE_TEMPERATURE = 0.7  # sampling temperature of the extra votes (vote 0 uses the run temperature)


class LLM:
    """OpenAI-compatible client (vLLM). The semaphore wraps single HTTP calls, never whole agents (no deadlock)."""

    def __init__(self, base_url, model, temperature=0.0, max_tokens=1024, concurrency=16, thinking=False):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(base_url=base_url, api_key="EMPTY", timeout=600)
        self.model, self.temperature, self.max_tokens, self.thinking = model, temperature, max_tokens, thinking
        self.sem = asyncio.Semaphore(concurrency)

    async def __call__(self, messages, tools=None, temperature=None) -> tuple[dict, int]:
        async with self.sem:
            r = await self.client.chat.completions.create(
                model=self.model, messages=messages, tools=tools,
                temperature=self.temperature if temperature is None else temperature,
                max_tokens=self.max_tokens, extra_body={"chat_template_kwargs": {"enable_thinking": self.thinking}})
        m = r.choices[0].message
        msg = {"role": "assistant", "content": m.content or ""}
        if m.tool_calls:
            msg["tool_calls"] = [{"id": c.id, "type": "function",
                                  "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                 for c in m.tool_calls]
        return msg, r.usage.completion_tokens


@dataclass
class Episode:
    row: pd.Series
    agents: list = field(default_factory=list)

    def add_agent(self, role, parent, messages, **extra) -> dict:
        agent = {"id": len(self.agents), "parent": parent, "role": role, "messages": messages, **extra}
        self.agents.append(agent)
        return agent


def parse_json(text: str):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    match = re.search(r"\{.*\}", text, flags=re.S)
    try:
        return json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        return None


async def run_subagent(llm, ep: Episode, parent, names, instructions):
    """Sub-agent call. With VOTES > 1, run that many independent extractions in parallel (vote 0 at the run
    temperature, the rest at VOTE_TEMPERATURE) and keep, per variable, the most common valid value
    (ties go to vote 0). Returns (values, errors, steps, tokens); steps/tokens count the slowest vote."""
    if VOTES == 1:
        return await extract(llm, ep, parent, names, instructions)
    outs = await asyncio.gather(*(extract(llm, ep, parent, names, instructions, vote=i,
                                          temperature=None if i == 0 else VOTE_TEMPERATURE) for i in range(VOTES)))
    values = {}
    for name in names:
        seen = [json.dumps(o[0][name]) for o in outs if name in o[0]]
        if seen:
            values[name] = json.loads(max(dict.fromkeys(seen), key=seen.count))  # first-seen wins ties
    errors = [] if len(values) == len(names) else outs[0][1]
    return values, errors, max(o[2] for o in outs), max(o[3] for o in outs)


async def extract(llm, ep: Episode, parent, names, instructions, vote=None, temperature=None):
    """One extraction agent with up to MAX_SUB_ATTEMPTS validation rounds. Returns (values, errors, steps, tokens)."""
    row = ep.row
    messages = prompts.subagent_messages(row, names, instructions)
    agent = ep.add_agent("subagent", parent, messages, variables=names, **({} if vote is None else {"vote": vote}))
    steps = tokens = 0
    for _ in range(MAX_SUB_ATTEMPTS):
        msg, n = await (llm(messages) if temperature is None else llm(messages, temperature=temperature))
        messages.append(msg)
        steps, tokens = steps + 1, tokens + n
        values, errors = validate(parse_json(msg["content"]), row["Calculator ID"], row["Patient Note"], names)
        if not errors:
            break
        messages.append(prompts.retry_message(errors))
    else:
        messages.pop()  # don't keep a dangling retry request
    agent.update(values=values, errors=errors)
    return values, errors, steps, tokens


def safe_calculate(calc_id, values):
    try:
        with redirect_stdout(io.StringIO()):
            return calculate(calc_id, values), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


async def run_orchestrator(llm, ep: Episode) -> dict:
    row, cid = ep.row, ep.row["Calculator ID"]
    spec_names = list(get_spec()[str(cid)])
    messages = prompts.orchestrator_messages(row)
    ep.add_agent("orchestrator", None, messages)
    values, steps, crit_tokens, requested = {}, 0, 0, set()
    for _ in range(MAX_ORCH_TURNS):
        msg, n = await llm(messages, prompts.TOOLS)
        messages.append(msg)
        steps, crit_tokens = steps + 1, crit_tokens + n
        calls = msg.get("tool_calls", [])
        if not calls:
            messages.append({"role": "user", "content": "Use the tools: spawn sub-agents, then call calculate."})
            continue
        jobs, results = [], {}
        for c in calls:
            if c["function"]["name"] != "spawn_subagent":
                continue
            try:
                args = json.loads(c["function"]["arguments"])
                names = [v for v in args["variables"] if v in spec_names]
                unknown = [v for v in args["variables"] if v not in spec_names]
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                results[c["id"]] = {"error": f"Invalid arguments: {e}"}
                continue
            if unknown or not names:
                results[c["id"]] = {"error": f"Unknown variable names: {unknown}. Use the exact names listed."}
                continue
            requested.update(names)
            jobs.append((c["id"], run_subagent(llm, ep, 0, names, args.get("instructions", ""))))
        outs = await asyncio.gather(*(j for _, j in jobs))
        for (cid_call, _), (vals, errs, s, t) in zip(jobs, outs):
            values.update(vals)
            results[cid_call] = {"values": vals, "errors": errs}
        if outs:  # one parallel stage: the slowest sub-agent sets its length
            steps += max(o[2] for o in outs)
            crit_tokens += max(o[3] for o in outs)
        answer = None
        for c in calls:
            if c["function"]["name"] == "calculate":
                answer, err = safe_calculate(cid, values)
                results[c["id"]] = {"answer": answer} if err is None else {"error": f"Calculation failed: {err}"}
            messages.append({"role": "tool", "tool_call_id": c["id"],
                             "content": prompts.tool_result(results.get(c["id"], {"error": "Unknown tool."}))})
        if answer is not None:
            return {"answer": answer, "values": values, "critical_steps": steps, "critical_tokens": crit_tokens,
                    "coverage": len(requested) / len(spec_names)}
    return {"answer": None, "values": values, "critical_steps": steps, "critical_tokens": crit_tokens,
            "coverage": len(requested) / len(spec_names)}


async def run_single(llm, ep: Episode) -> dict:
    """Control: one extraction agent gets all variables at once, same prompt format and calculate()."""
    cid = ep.row["Calculator ID"]
    names = list(get_spec()[str(cid)])
    values, errors, steps, tokens = await run_subagent(llm, ep, None, names, "Extract all variables.")
    answer, _ = safe_calculate(cid, values)
    return {"answer": answer, "values": values, "critical_steps": steps + 1, "critical_tokens": tokens, "coverage": 1.0}


async def run_cot(llm, ep: Episode) -> dict:
    messages = prompts.cot_messages(ep.row)
    ep.add_agent("cot", None, messages)
    msg, n = await llm(messages)
    messages.append(msg)
    nums = re.findall(r"Answer:\s*\**\s*(-?\d+(?:\.\d+)?)", msg["content"])
    return {"answer": float(nums[-1]) if nums else None, "values": {}, "critical_steps": 1, "critical_tokens": n,
            "coverage": None}


MODES = {"orchestrator": run_orchestrator, "single": run_single, "cot": run_cot}


async def run_episode(llm, mode, row) -> dict:
    ep, t0 = Episode(row), time.perf_counter()
    try:
        out, error = await MODES[mode](llm, ep), None
    except Exception as e:  # keep going on server hiccups; the row is logged with the error
        out, error = {"answer": None, "values": {}}, f"{type(e).__name__}: {e}"
    return {"row": int(row["Row Number"]), "calc_id": int(row["Calculator ID"]), "calc": row["Calculator Name"],
            "note_type": row["Note Type"], "mode": mode, "model": llm.model, "gold": str(row["Ground Truth Answer"]),
            "correct": is_correct(out["answer"], row), "oracle_consistent": bool(row["oracle_consistent"]),
            "excluded": row["Calculator Name"] in EXCLUDED, "wall_s": time.perf_counter() - t0,
            "n_subagents": sum(a["role"] == "subagent" and a.get("vote", 0) == 0 for a in ep.agents), "error": error,
            "gold_entities": row["entities"], **out, "agents": ep.agents}


def select_rows(split, limit=None, generation=False) -> pd.DataFrame:
    df = load(split)
    flags = pd.read_parquet(ROOT / "data" / "oracle" / f"{split}.parquet")[["Row Number", "oracle_consistent"]]
    df = df.merge(flags, on="Row Number")
    if generation:  # teacher data: only rows whose label the official code reproduces
        df = df[df["oracle_consistent"] & ~df["Calculator Name"].isin(EXCLUDED)]
    # interleave calculators (seeded): any prefix, from --limit or a time-capped run, is stratified
    df = df.sample(frac=1, random_state=0)
    df = df.iloc[df.groupby("Calculator Name").cumcount().argsort(kind="stable")]
    return df.head(limit) if limit else df


async def main_async(a):
    global VOTES
    VOTES = a.votes
    llm = LLM(a.base_url, a.model, a.temperature, a.max_tokens, a.concurrency, a.thinking)
    df = select_rows(a.split, a.limit, a.generation)
    if a.only_failed:  # rejection sampling: retry rows with no correct episode in these logs
        ok = {r["row"] for f in a.only_failed for l in open(f, encoding="utf-8") if (r := json.loads(l))["correct"]}
        df = df[~df["Row Number"].astype(int).isin(ok)]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["row"] for l in out.open(encoding="utf-8")} if out.exists() else set()
    todo = [r for _, r in df.iterrows() if int(r["Row Number"]) not in done]
    print(f"{len(todo)} episodes to run ({len(done)} already in {out})")
    n_ok, slots = 0, asyncio.Semaphore(a.episodes)  # bounded: episodes finish steadily, wall_s excludes queueing

    async def bounded(r):
        async with slots:
            return await run_episode(llm, a.mode, r)

    with out.open("a", encoding="utf-8") as f:
        for i, fut in enumerate(asyncio.as_completed([asyncio.create_task(bounded(r)) for r in todo]), 1):
            rec = await fut
            n_ok += rec["correct"]
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} acc={n_ok / i:.1%}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--limit", type=int)
    p.add_argument("--generation", action="store_true", help="only oracle-consistent, non-excluded rows")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--concurrency", type=int, default=16, help="max LLM calls in flight")
    p.add_argument("--episodes", type=int, default=8, help="max episodes in flight")
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--votes", type=int, default=1, help="extractions per sub-agent call (majority vote)")
    p.add_argument("--only-failed", nargs="+", metavar="LOG", help="only rows with no correct episode in these logs")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
