#!/usr/bin/env bash
# Large-scale bit-exactness check: the same C kernels compiled for the host
# (custom instructions emulated in C) must reproduce the Python integer
# reference logits on N ImageNet val images, for every kernel configuration.
#   MODEL=sparse N=1000 scripts/check_native.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MODEL="${MODEL:-sparse}"; N="${N:-1000}"; PY="${PY:-.venv/bin/python}"
DIR="models/$MODEL/export_check$N"
[ -f "$DIR/golden.bin" ] || $PY python/export_c.py "models/$MODEL/qmodel.pt" --out "$DIR" --nimg "$N" --seed 1
fail=0
for c in ${CFGS:-B1 X2 S0 S1 S2}; do
  make -s native MODEL=$MODEL CFG=$c
  out=$(build/native/$MODEL/${c}_n1_g8_d1001/resnet18 "$DIR" | grep RESULT)
  echo "$MODEL $c: $out"
  echo "$out" | grep -q PASS || fail=1
done
exit $fail
