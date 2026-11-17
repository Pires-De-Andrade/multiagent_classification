# medswarm — distilling a parallel multi-agent orchestrator into small models

A large Qwen teacher acts as an **orchestrator** on MedCalc-Bench rule-based calculators: it splits a
calculator's variables into groups and spawns **parallel extraction sub-agents** (`spawn_subagent`),
then calls `calculate` (the official MedCalc-Bench Python implementation). Correct teacher episodes
are distilled (LoRA SFT, assistant-only loss) into a 2B / 4B Qwen student, which then plays both roles.

Background and evidence: `reports/Plano orquestrador MedCalc destilado.md`.
**Research prototype, not for clinical use.**

## Results (MedCalc-Bench Verified test, 17 calculators, n=340)

Teacher: Qwen3.6-35B-A3B (AWQ). Students: Qwen3.5-2B / 4B, LoRA SFT on correct teacher episodes.
`var_acc`: share of annotated variables extracted correctly. `crit. tokens`: tokens on the critical path
(parallel sub-agents counted once).

| Condition | Acc. | var_acc | No answer | Crit. tokens |
|---|---|---|---|---|
| B1: Qwen3.5-2B, single call, no training | 9.4% | 52.3% | 68.5% | 597 |
| M: 2B student | 71.8% | 91.4% | 6.8% | 693 |
| M2: 2B student + rejection-sampled data | 75.9% | 92.5% | 4.1% | 665 |
| **M4: 4B student** (same data as M2) | **80.9%** | 93.2% | 1.2% | 647 |
| M4 + 5-vote sub-agents | 81.8% | 93.4% | 0.0% | 703 |
| M4 + GRPO on the sub-agent role | 75.3% | 94.4% | 0.3% | 528 |
| T1: teacher, orchestrator | 79.1% | 92.8% | 5.6% | 1561 |
| T2: teacher, single agent | 58.8% | 88.0% | 20.3% | 666 |

- The 4B student matches the 35B teacher (M4 vs T1: +1.8 pts, McNemar p=0.53) with ~2.4x fewer critical-path tokens.
- Orchestration helps the teacher mostly through robustness: T2 gives no answer in 20% of cases (T1 vs T2: +20 pts).
- GRPO is a negative result: the reward (per-variable match against GPT-4 train extractions) went up, case
  accuracy went down. False-positive booleans almost doubled and errors on variables outside the annotation
  (invisible to both the reward and `var_acc`) rose 71 -> 123. Reward the final calculator output instead.
- Not isolated: the effect of coordination at a fixed student size (a trained single-agent 2B/4B control was not run).

## Data facts (verified locally)

- MedCalc-Bench Verified, HF commit `58fc3de5`: rule-based = 2,426 train / 380 test rows, 19 calculators, all with train data.
- Oracle (`calculate(Relevant Entities)` vs label): 96.4% train / 93.7% test consistent. Labels were produced by the
  same code, so this validates the pipeline, not clinical correctness.
- APACHE II and Caprini are excluded from the main metric (corrupted labels; oracle 50% / 45% on test) and reported separately.

## Layout

| File | Role |
|---|---|
| `src/medswarm/data.py` | load Verified, rule-based subset, exclusions |
| `src/medswarm/calculators.py` | `calculate()` over the official implementations, official correctness rule |
| `src/medswarm/oracle.py` | O0 oracle → `data/oracle/*.parquet` (consistency flags) |
| `src/medswarm/spec.py` | variable spec per calculator + extraction validator (types, units, verbatim evidence) |
| `src/medswarm/prompts.py` | prompts and tool schemas |
| `src/medswarm/harness.py` | async episodes: `orchestrator`, `single` (control), `cot`; voting; JSONL logs, resumable |
| `src/medswarm/build_sft.py` | filter teacher episodes → chat samples (orchestrator tool calls + sub-agent answers) |
| `src/medswarm/train_sft.py` | LoRA SFT (TRL), mask check, merge |
| `src/medswarm/train_grpo.py` | GRPO on the sub-agent role (TRL), resume, merge |
| `src/medswarm/evaluate.py` | accuracy, per-variable accuracy, critical steps, McNemar + bootstrap + Holm |
| `src/medswarm/demo.py` | agent tree of one episode (log or live) |
| `scripts/remote.sh`, `setup_server.sh`, `serve.sh` | server control |

## Local (no GPU)

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"
python -m medswarm.oracle      # writes data/oracle/, prints consistency per calculator
pytest -q                      # calculate/oracle, validator, orchestrator + single episodes with a fake LLM
```

## Server footprint and uninstall

Everything lives inside the project dir: `scripts/env.sh` points uv, Python installs, HF, vLLM, Triton and
torch caches at `.cache/`. Run `source scripts/env.sh` before any manual command (the scripts do it themselves).

| Path | Contents |
|---|---|
| `.venv-serve/`, `.venv-train/` | the two venvs (vLLM / training) |
| `.cache/` | uv packages + Python 3.12, HF models, compile caches |
| `data/`, `third_party/` | dataset + official calculator code (`scripts/fetch_data.sh`) |
| `runs/`, `logs/` | episodes, checkpoints, logs: **copy/upload before deleting** |

Uninstall: `pkill -f 'vllm serve'; rm -rf .venv-serve .venv-train .cache data third_party runs logs`.

Check the sha256 of downloaded model shards against the Hub: a silently corrupted shard once produced a
model that trained at loss = ln(vocab) and generated only "!".

## GPU runbook (server, one model on the GPU at a time)

Developed on one RTX 4090 (24 GB). vLLM uses TRITON_ATTN and the PyTorch sampler, so no CUDA toolkit is needed.

```bash
# 0. once: key auth, then setup
export HOST=user@server
ssh-copy-id $HOST
scripts/remote.sh check && scripts/remote.sh sync
scripts/remote.sh run setup "bash scripts/setup_server.sh"

PY=.venv-serve/bin/python; TR=.venv-train/bin/python
T=hyperhuzaifa/Qwen3.6-35B-A3B-W4A16-AWQ

# 1. teacher data on train (resumable: rerun the same command to continue), then two
#    rejection-sampling passes on the rows the teacher still gets wrong
scripts/remote.sh run teacher "scripts/serve.sh $T"
scripts/remote.sh run gen "$PY -m medswarm.harness --mode orchestrator --split train --generation --temperature 0.7 --model $T --out runs/teacher_train.jsonl"
scripts/remote.sh run r2 "$PY -m medswarm.harness --mode orchestrator --split train --generation --temperature 0.8 --model $T --only-failed runs/teacher_train.jsonl --out runs/teacher_train_r2.jsonl"
scripts/remote.sh run r3 "$PY -m medswarm.harness --mode orchestrator --split train --generation --temperature 0.8 --model $T --only-failed runs/teacher_train.jsonl runs/teacher_train_r2.jsonl --out runs/teacher_train_r3.jsonl"

# 2. teacher on test: T1 orchestrator, T2 single agent
scripts/remote.sh run t1 "$PY -m medswarm.harness --mode orchestrator --model $T --out runs/T1.jsonl"
scripts/remote.sh run t2 "$PY -m medswarm.harness --mode single --model $T --out runs/T2.jsonl"

# 3. untrained student baselines (stop the teacher first: pkill -f 'bin/vllm serve')
S=Qwen/Qwen3.5-4B                                # the B1 row in the table used Qwen/Qwen3.5-2B
scripts/remote.sh run student "scripts/serve.sh $S"
for m in cot:B0 single:B1 orchestrator:B2; do
  scripts/remote.sh run ${m#*:} "$PY -m medswarm.harness --mode ${m%:*} --model $S --out runs/${m#*:}.jsonl"
done

# 4. SFT (stop vLLM first); --check-mask prints which tokens are trained
TPL=qwen3_5_nothink_training.jinja               # resolved inside the trl package on the server
scripts/remote.sh run build "$TR -m medswarm.build_sft runs/teacher_train*.jsonl --out data/sft"
scripts/remote.sh run mask  "$TR -m medswarm.train_sft --model $S --chat-template $TPL --data data/sft --out runs/sft --check-mask"
scripts/remote.sh run train "$TR -m medswarm.train_sft --model $S --chat-template $TPL --data data/sft --out runs/sft --epochs 1 --liger"

# 5. distilled student (inference uses the base model's chat template, not the training one)
cp .cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/*/chat_template.jinja runs/sft/merged/
scripts/remote.sh run student "scripts/serve.sh runs/sft/merged"
scripts/remote.sh run m "$PY -m medswarm.harness --mode orchestrator --model runs/sft/merged --out runs/M4.jsonl"
scripts/remote.sh run mv "$PY -m medswarm.harness --mode orchestrator --model runs/sft/merged --votes 5 --concurrency 32 --out runs/M4_vote5.jsonl"

# 6. optional GRPO on the sub-agent role (negative result above); --micro-batch/--gen-prompts fit 24 GB
scripts/remote.sh run grpo "$TR -m medswarm.train_grpo --model runs/sft/merged --out runs/grpo --micro-batch 2 --gen-prompts 1 --resume"
scripts/remote.sh run merge "$TR -m medswarm.train_grpo --model runs/sft/merged --out runs/grpo --merge runs/grpo/checkpoint-N"

# 7. back on the laptop
scripts/remote.sh pull
python -m medswarm.evaluate B1=runs/B1.jsonl M4=runs/M4.jsonl M4_vote5=runs/M4_vote5.jsonl T1=runs/T1.jsonl T2=runs/T2.jsonl --baseline M4
python -m medswarm.demo runs/M4.jsonl --row <row>
```
