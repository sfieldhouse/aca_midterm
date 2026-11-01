#!/usr/bin/env bash
# End-to-end reproduction: toolchain/Spike, models (fine-tune + quantize + export),
# tests, Spike benchmarks, figures.  Steps whose outputs exist are skipped.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-.venv/bin/python}"
EVAL_GPU="${EVAL_GPU:-0}"
[ -x tools/riscv/bin/riscv64-unknown-elf-gcc ] || scripts/setup_toolchain.sh
[ -x tools/spike/bin/spike ] || scripts/build_spike.sh
scripts/train_sparse.sh
for m in dense:models/resnet18_tv_in1k.safetensors sparse:models/sparse/model.pt; do
  n=${m%%:*}; c=${m#*:}
  mkdir -p models/$n
  sc=""; [ -f models/$n/qat_scales.json ] && sc="--scales models/$n/qat_scales.json"   # QAT grid
  [ -f models/$n/qmodel.pt ] || CUDA_VISIBLE_DEVICES=$EVAL_GPU $PY python/quantize.py "$c" $sc \
      --out models/$n/qmodel.pt --json results/accuracy/int8_${n}_val.json
  [ -f models/$n/export/model.bin ] || CUDA_VISIBLE_DEVICES=$EVAL_GPU $PY python/export_c.py \
      models/$n/qmodel.pt --out models/$n/export --nimg 16
done
$PY python/plot_sparsity.py
MODEL=sparse scripts/run_tests.sh
$PY scripts/run_benchmarks.py --nimg 8 --mix --sweep
$PY scripts/make_figures.py
