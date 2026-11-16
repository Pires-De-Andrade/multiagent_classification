"""LoRA SFT of the student on teacher trajectories (server only, needs the [train] extras).

Loss only on assistant tokens (tool results and prompts are masked). Run with --check-mask first:
it prints exactly which tokens of one orchestrator sample would be trained on, then exits.

Usage:
  python -m medswarm.train_sft --model Qwen/Qwen3.5-2B --out runs/sft_qwen35_2b --check-mask
  python -m medswarm.train_sft --model Qwen/Qwen3.5-2B --out runs/sft_qwen35_2b
"""
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--data", default="data/sft")
    p.add_argument("--chat-template", help="training template with {% generation %} markers (e.g. TRL's qwen3_5_nothink_training.jinja)")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--check-mask", action="store_true")
    a = p.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    if a.chat_template:  # a path, or a template file name shipped inside the trl package
        path = Path(a.chat_template)
        if not path.exists():
            import trl
            path = next(Path(trl.__file__).parent.rglob(a.chat_template))
        tok.chat_template = path.read_text()
    ds = load_dataset("json", data_files={"train": f"{a.data}/train.jsonl", "val": f"{a.data}/val.jsonl"})
    ds = ds.remove_columns("kind")

    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
    cfg = SFTConfig(
        output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine",
        warmup_ratio=0.03, per_device_train_batch_size=2, gradient_accumulation_steps=8,
        gradient_checkpointing=True, bf16=True, max_length=a.max_length, assistant_only_loss=True,
        logging_steps=10, eval_strategy="epoch", save_strategy="epoch", report_to="none")
    lora = LoraConfig(r=a.rank, lora_alpha=a.rank, target_modules="all-linear", task_type="CAUSAL_LM")
    # processing_class=tok (not a processor) avoids TRL's "assistant-only loss not supported for VLMs" (trl#5471)
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds["train"], eval_dataset=ds["val"],
                         processing_class=tok, peft_config=lora)

    if a.check_mask:
        ex = next(e for e in trainer.train_dataset if sum(e["assistant_masks"]) < len(e["input_ids"]))
        trained = [t for t, m in zip(ex["input_ids"], ex["assistant_masks"]) if m]
        print(f"{sum(ex['assistant_masks'])}/{len(ex['input_ids'])} tokens trained. Trained text:\n")
        print(tok.decode(trained))
        lengths = sorted(len(e["input_ids"]) for e in trainer.train_dataset)
        print(f"\nlengths: median={lengths[len(lengths) // 2]} max={lengths[-1]} (max_length={a.max_length})")
        return

    trainer.train()
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(f"{a.out}/merged")
    tok.save_pretrained(f"{a.out}/merged")
    print(f"merged model saved to {a.out}/merged")


if __name__ == "__main__":
    main()
