"""Prompts and tool schemas for the orchestrator, extraction sub-agents and the CoT baseline."""
import json

from medswarm.spec import describe, get_spec

ORCHESTRATOR_SYSTEM = """You are the orchestrator of a team of clinical extraction sub-agents.
Goal: compute a medical score for the patient note. You never extract values yourself.
1. Split the calculator's variables into small groups of related variables (e.g. vital signs, labs, history) and call spawn_subagent once per group. Call several in the same turn: they run in parallel. Every variable should be covered by exactly one group.
2. Read the results. If a sub-agent failed on some variables, you may spawn again for just those.
3. Call calculate to finish. Variables that are not mentioned in the note are scored as absent."""

SUBAGENT_SYSTEM = """You extract clinical variables from a patient note.
Answer with a single JSON object and nothing else, with one key per requested variable:
{"<variable name>": {"value": <value or null>, "evidence": "<verbatim quote from the note>"}}
Rules:
- value must follow the requested format exactly.
- Use null when the note does not mention the variable (evidence may then be "").
- Use false for a condition only when the note states it is absent.
- evidence must be copied word for word from the note."""

COT_SYSTEM = "You are a careful clinician. Think step by step, then end with a final line 'Answer: <number>'."

TOOLS = [
    {"type": "function", "function": {
        "name": "spawn_subagent",
        "description": "Start a sub-agent that reads the note and extracts the given variables. Returns their values.",
        "parameters": {"type": "object", "properties": {
            "variables": {"type": "array", "items": {"type": "string"}, "description": "Exact variable names."},
            "instructions": {"type": "string", "description": "Short hints for this group (what to look for)."},
        }, "required": ["variables", "instructions"]}}},
    {"type": "function", "function": {
        "name": "calculate",
        "description": "Run the official calculator on all values extracted so far and finish.",
        "parameters": {"type": "object", "properties": {}}}},
]


def variable_list(calc_id: int, names: list[str]) -> str:
    spec = get_spec()[str(calc_id)]
    return "\n".join(f"- {n}: {describe(spec[n])}" for n in names)


def orchestrator_messages(row) -> list[dict]:
    names = list(get_spec()[str(row["Calculator ID"])])
    user = (f"Calculator: {row['Calculator Name']}\nQuestion: {row['Question']}\n\n"
            f"Variables:\n{variable_list(row['Calculator ID'], names)}\n\nPatient note:\n{row['Patient Note']}")
    return [{"role": "system", "content": ORCHESTRATOR_SYSTEM}, {"role": "user", "content": user}]


def subagent_messages(row, names: list[str], instructions: str) -> list[dict]:
    user = (f"Patient note:\n{row['Patient Note']}\n\nInstructions: {instructions}\n\n"
            f"Extract these variables:\n{variable_list(row['Calculator ID'], names)}")
    return [{"role": "system", "content": SUBAGENT_SYSTEM}, {"role": "user", "content": user}]


def retry_message(errors: list[str]) -> dict:
    return {"role": "user", "content": "Fix these problems and answer again with the full JSON object:\n"
            + "\n".join(f"- {e}" for e in errors)}


def cot_messages(row) -> list[dict]:
    return [{"role": "system", "content": COT_SYSTEM},
            {"role": "user", "content": f"{row['Patient Note']}\n\n{row['Question']}"}]


def tool_result(payload) -> str:
    return json.dumps(payload, ensure_ascii=False)
