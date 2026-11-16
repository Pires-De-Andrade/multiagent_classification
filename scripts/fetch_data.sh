#!/usr/bin/env bash
# Fetch the git-ignored inputs at the exact versions used during development, then build the oracle flags.
#   PYTHON=.venv-train/bin/python bash scripts/fetch_data.sh
set -euo pipefail
HF_COMMIT=58fc3de568bfc53a68f32327e58f60dc6eaa17f8   # nsk7153/MedCalc-Bench-Verified
MC_COMMIT=20b10f9d66a8c3b0356ae3138533f5ec91666b6a   # ncbi-nlp/MedCalc-Bench (official calculator code)

mkdir -p data/raw third_party
for f in train test one_shot; do
  curl -fsSL -o "data/raw/$f.parquet" \
    "https://huggingface.co/datasets/nsk7153/MedCalc-Bench-Verified/resolve/$HF_COMMIT/data/$f-00000-of-00001.parquet"
done
[ -d third_party/MedCalc-Bench ] || git clone -q --filter=blob:none https://github.com/ncbi-nlp/MedCalc-Bench.git third_party/MedCalc-Bench
git -C third_party/MedCalc-Bench checkout -q "$MC_COMMIT"

"${PYTHON:-python}" -m medswarm.oracle   # expect 96.4% train / 93.7% test consistent
