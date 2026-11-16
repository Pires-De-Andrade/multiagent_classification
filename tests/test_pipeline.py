import asyncio
import json
import re

import pytest

from medswarm.calculators import calculate, is_correct
from medswarm.harness import run_episode, select_rows
from medswarm.spec import get_spec, validate


@pytest.fixture(scope="module")
def row():
    df = select_rows("train", generation=True)
    return df[df["Calculator ID"] == 4].iloc[0]  # CHA2DS2-VASc


class FakeLLM:
    """Scripted model: orchestrator spawns two groups then calculates; sub-agents answer with gold values."""
    model = "fake"

    def __init__(self, row):
        self.row = row

    async def __call__(self, messages, tools=None):
        if tools:  # orchestrator
            if messages[-1]["role"] == "user":
                names = list(get_spec()[str(self.row["Calculator ID"])])
                half = len(names) // 2
                calls = [self._call(f"c{i}", "spawn_subagent", {"variables": g, "instructions": "x"})
                         for i, g in enumerate([names[:half], names[half:]])]
            else:
                calls = [self._call("c9", "calculate", {})]
            return {"role": "assistant", "content": "", "tool_calls": calls}, 5
        requested = re.findall(r"^- (.+?): ", messages[1]["content"].split("Extract these variables:")[1], re.M)
        quote = self.row["Patient Note"][:30]
        out = {n: {"value": self.row["entities"].get(n), "evidence": quote} for n in requested}
        return {"role": "assistant", "content": json.dumps(out)}, 7

    @staticmethod
    def _call(cid, name, args):
        return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def test_calculate_reproduces_label(row):
    assert is_correct(calculate(row["Calculator ID"], row["entities"]), row)


def test_validator_rejects_bad_values(row):
    note = row["Patient Note"]
    names = list(get_spec()["4"])
    bad = {n: {"value": None, "evidence": ""} for n in names}
    bad["Hypertension history"] = {"value": "yes", "evidence": note[:20]}
    bad["Stroke"] = {"value": True, "evidence": "not in the note at all"}
    _, errors = validate(bad, 4, note, names)
    assert len(errors) == 2
    assert validate("nope", 4, note, names)[1]


def test_orchestrator_episode(row):
    rec = asyncio.run(run_episode(FakeLLM(row), "orchestrator", row))
    assert rec["error"] is None and rec["correct"]
    assert rec["n_subagents"] == 2 and rec["coverage"] == 1.0
    assert rec["critical_steps"] == 3  # spawn turn + parallel sub-agents + calculate turn
    assert rec["critical_tokens"] == 5 + 7 + 5
    assert [a["parent"] for a in rec["agents"]] == [None, 0, 0]


def test_single_agent_episode(row):
    rec = asyncio.run(run_episode(FakeLLM(row), "single", row))
    assert rec["error"] is None and rec["correct"] and rec["n_subagents"] == 1
