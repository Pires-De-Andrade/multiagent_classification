"""GRPO on the extraction sub-agent role (single turn), starting from an SFT-merged student.

Why only the sub-agent: 71% of the student's wrong variables are unanimous across 5 votes, i.e. systematic
extraction errors. SFT can only imitate the teacher; RL with a verifiable reward can fix errors the
teacher also makes. The orchestrator is left as SFT (multi-agent RL is future work).

Prompts: sub-agent calls from teacher episodes on train (same variable groupings the orchestrator makes),
including rows the teacher got wrong. Reward: fraction of requested variables that are valid (spec
validator, verbatim evidence) and match the gold Relevant Entities; 0 if the output is not JSON.
Caveat: train entities are GPT-4 extractions, not human-verified, so the reward is noisy.

Usage:
  python -m medswarm.train_grpo --model runs/sft_qwen35_r2/merged --out runs/grpo --max-steps 10   # smoke test
  python -m medswarm.train_grpo --model runs/sft_qwen35_r2/merged --out runs/grpo                  # long run
  python -m medswarm.train_grpo --model runs/sft_qwen35_r2/merged --out runs/grpo --merge runs/grpo/checkpoint-N
"""
import argparse
import json
import random

import torch
from datasets import Dataset

from medswarm.evaluate import same_value
from medswarm.harness import parse_json
from medswarm.spec import get_spec, validate


def build_prompts(logs: list[str], seed=0) -> Dataset:
    seen, rows = set(), []
    for f in logs:
        for line in open(f, encoding="utf-8"):
            rec = json.loads(line)
            if not rec["oracle_consistent"] or rec["excluded"]:
                continue
            for a in rec["agents"]:
                if a["role"] != "subagent":
                    continue
                key = (rec["row"], tuple(a["variables"]))
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"prompt": a["messages"][:2], "names": a["variables"], "calc_id": rec["calc_id"],
                             "gold": json.dumps(rec["gold_entities"]), "note": a["messages"][1]["content"]})
    random.Random(seed).shuffle(rows)
    return Dataset.from_list(rows)


def extraction_reward(completions, names, calc_id, gold, note, **_):
    """Per completion: share of requested variables that are valid and equal to gold."""
    out = []
    for comp, ns, cid, g, nt in zip(completions, names, calc_id, gold, note):
        text = comp[0]["content"] if isinstance(comp, list) else comp
        parsed = parse_json(text)
        if parsed is None:
            out.append(0.0)
            continue
        values, _ = validate(parsed, cid, nt, ns)  # invalid variables are dropped
        g, spec = json.loads(g), get_spec()[str(cid)]
        hits = sum(n in values and same_value(values[n], g.get(n), spec[n]["type"]) for n in ns)
        out.append(hits / len(ns))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="SFT-merged student")
    p.add_argument("--out", required=True)
    p.add_argument("--logs", nargs="+", default=["runs/teacher_train.jsonl"])
    p.add_argument("--max-steps", type=int, default=10_000)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--generations", type=int, default=8)
    p.add_argument("--prompts-per-step", type=int, default=4)
    p.add_argument("--merge", help="checkpoint dir: merge its LoRA into --model, save to OUT/merged, exit")
    a = p.parse_args()

    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    tok = AutoTokenizer.from_pretrained(a.model)
    if a.merge:
        base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
        merged = PeftModel.from_pretrained(base, a.merge).merge_and_unload()
        merged.save_pretrained(f"{a.out}/merged")
        tok.save_pretrained(f"{a.out}/merged")
        print(f"merged {a.merge} -> {a.out}/merged")
        return

    ds = build_prompts(a.logs)
    print(f"{len(ds)} sub-agent prompts")
    cfg = GRPOConfig(
        output_dir=a.out, max_steps=a.max_steps, learning_rate=a.lr, lr_scheduler_type="constant_with_warmup",
        warmup_steps=10, per_device_train_batch_size=a.generations, gradient_accumulation_steps=a.prompts_per_step,
        num_generations=a.generations, max_completion_length=384, temperature=1.0, beta=0.0,
        chat_template_kwargs={"enable_thinking": False}, gradient_checkpointing=True, bf16=True,
        logging_steps=1, save_steps=a.save_steps, save_total_limit=20, report_to="none", log_completions=False)
    lora = LoraConfig(r=a.rank, lora_alpha=16, use_rslora=True, lora_dropout=0.0,
                      target_modules="all-linear", task_type="CAUSAL_LM")
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
    trainer = GRPOTrainer(model=model, reward_funcs=extraction_reward, args=cfg, train_dataset=ds,
                          processing_class=tok, peft_config=lora)
    trainer.train()
    print(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
