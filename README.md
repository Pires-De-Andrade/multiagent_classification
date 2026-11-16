# medswarm — distilling a parallel multi-agent orchestrator into a 2B model

A large Qwen teacher acts as an **orchestrator** on MedCalc-Bench rule-based calculators: it splits a
calculator's variables into groups and spawns **parallel extraction sub-agents** (`spawn_subagent`),
then calls `calculate` (the official MedCalc-Bench Python implementation). Correct teacher episodes
are distilled (LoRA SFT, assistant-only loss) into a ~2B Qwen student, which then plays both roles.

Question: does learned coordination beat the same 2B model extracting everything in one call?
Background and evidence: `reports/Plano orquestrador MedCalc destilado.md`. Research prototype, not for clinical use.

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
| `src/medswarm/harness.py` | async episodes: `orchestrator`, `single` (control), `cot`; JSONL logs, resumable |
| `src/medswarm/build_sft.py` | filter teacher episodes → chat samples |
| `src/medswarm/train_sft.py` | LoRA SFT (TRL), mask check, merge |
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

## GPU runbook (server, one model on the GPU at a time)

```bash
# 0. once: key auth, then setup
ssh-copy-id user_joaoandrade@10.100.60.21
scripts/remote.sh check && scripts/remote.sh sync
scripts/remote.sh run setup "bash scripts/setup_server.sh"

PY=.venv-serve/bin/python
T=hyperhuzaifa/Qwen3.6-35B-A3B-W4A16-AWQ        # fallback: stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ

# 1. teacher pilot (decides the Qwen generation): tool-call validity, accuracy, throughput
scripts/remote.sh run teacher "scripts/serve.sh $T"
scripts/remote.sh run pilot "$PY -m medswarm.harness --mode orchestrator --split train --generation --limit 50 --temperature 0.7 --model $T --out runs/pilot_teacher.jsonl"

# 2. data generation (resumable: rerun the same command to continue)
scripts/remote.sh run gen "$PY -m medswarm.harness --mode orchestrator --split train --generation --temperature 0.7 --model $T --out runs/teacher_train.jsonl"

# 3. teacher on test: T1 orchestrator, T2 single agent
scripts/remote.sh run t1 "$PY -m medswarm.harness --mode orchestrator --model $T --out runs/T1.jsonl"
scripts/remote.sh run t2 "$PY -m medswarm.harness --mode single --model $T --out runs/T2.jsonl"

# 4. student baselines (stop the teacher first: pkill -f 'vllm serve')
S=Qwen/Qwen3.5-2B                                # fallback: Qwen/Qwen3-1.7B
scripts/remote.sh run student "scripts/serve.sh $S"
for m in cot:B0 single:B1 orchestrator:B2; do
  scripts/remote.sh run ${m#*:} "$PY -m medswarm.harness --mode ${m%:*} --model $S --out runs/${m#*:}.jsonl"
done

# 5. SFT (stop vLLM first)
TPL=qwen3_5_nothink_training.jinja               # resolved inside the trl package on the server
scripts/remote.sh run build ".venv-train/bin/python -m medswarm.build_sft runs/teacher_train.jsonl --out data/sft"
scripts/remote.sh run mask  ".venv-train/bin/python -m medswarm.train_sft --model $S --chat-template $TPL --out runs/sft_qwen35 --check-mask"
scripts/remote.sh run train ".venv-train/bin/python -m medswarm.train_sft --model $S --chat-template $TPL --out runs/sft_qwen35"

# 6. distilled student
scripts/remote.sh run student "scripts/serve.sh runs/sft_qwen35/merged"
scripts/remote.sh run m "$PY -m medswarm.harness --mode orchestrator --model runs/sft_qwen35/merged --out runs/M.jsonl"

# 7. back on the laptop
scripts/remote.sh pull
python -m medswarm.evaluate B0=runs/B0.jsonl B1=runs/B1.jsonl B2=runs/B2.jsonl M=runs/M.jsonl T1=runs/T1.jsonl T2=runs/T2.jsonl --baseline B1
python -m medswarm.demo runs/M.jsonl --row <row>
```

Go/no-go: switch to the fallback teacher if the pilot shows malformed tool calls or loops; switch to
Qwen3-1.7B if `--check-mask` shows anything other than assistant turns being trained.
