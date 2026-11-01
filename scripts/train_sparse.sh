#!/usr/bin/env bash
# Kurtz et al. (2020) activation-sparsity pipeline for ResNet-18 on ImageNet:
#   1. fine-tune with square-Hoyer activation regularisation
#   2. per-site FATReLU sensitivity analysis (static thresholds)
#   3. recovery fine-tune with FATReLU thresholds fixed (THSCALE x the
#      sensitivity thresholds), Hoyer off, self-distillation from the phase-1
#      model and quantization-aware training on the W8A8 integer grid
#   4. FP32 val accuracy + per-layer sparsity of baseline / Hoyer / final model
# Requires 3 GPUs (DDP); set GPUS / NPROC to change.  ~2.5 h on 3x RTX A5000.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-.venv/bin/python}"
GPUS="${GPUS:-0,1,2}"
NPROC="${NPROC:-3}"
LAMBDA="${LAMBDA:-3e-9}"     # phase 1 Hoyer strength
LR2="${LR2:-0.003}"          # phase 3 peak LR
THSCALE="${THSCALE:-0.85}"   # scale applied to the sensitivity thresholds
TOL="${TOL:-0.005}"          # per-site CE tolerance of the sensitivity analysis
EP1="${EP1:-10}"
EP2="${EP2:-6}"
EVAL_GPU="${EVAL_GPU:-${GPUS%%,*}}"
RUN="CUDA_VISIBLE_DEVICES=$GPUS $PY -m torch.distributed.run --nproc_per_node=$NPROC"
mkdir -p models results/accuracy results/sensitivity

[ -f models/resnet18_tv_in1k.safetensors ] || $PY python/download_model.py

if [ ! -f models/hoyer/model.pt ]; then
  eval $RUN python/finetune_sparse.py --init models/resnet18_tv_in1k.safetensors \
       --hoyer "$LAMBDA" --epochs "$EP1" --out models/hoyer
fi

if [ ! -f results/sensitivity/hoyer_thresholds.json ]; then
  CUDA_VISIBLE_DEVICES=$EVAL_GPU $PY python/sensitivity.py models/hoyer/model.pt \
       --tol "$TOL" --out results/sensitivity/hoyer.json
fi

if [ ! -f models/sparse/model.pt ]; then
  $PY -c "import json; t=json.load(open('results/sensitivity/hoyer_thresholds.json')); \
json.dump({k: v * $THSCALE for k, v in t.items()}, open('results/sensitivity/final_thresholds.json', 'w'), indent=1)"
  eval $RUN python/finetune_sparse.py --init models/hoyer/model.pt \
       --thresholds results/sensitivity/final_thresholds.json --hoyer 0 --lr "$LR2" --epochs "$EP2" \
       --qat --kd_teacher models/hoyer/model.pt --kd_alpha 0.7 --kd_T 2 --out models/sparse
fi

for m in baseline:models/resnet18_tv_in1k.safetensors hoyer:models/hoyer/model.pt sparse:models/sparse/model.pt; do
  n=${m%%:*}; c=${m#*:}
  CUDA_VISIBLE_DEVICES=$EVAL_GPU $PY python/eval_fp32.py "$c" --json results/accuracy/fp32_${n}_val.json
  CUDA_VISIBLE_DEVICES=$EVAL_GPU $PY python/measure_sparsity.py --name "$n" --ckpt "$c"
done
