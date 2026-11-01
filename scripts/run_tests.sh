#!/usr/bin/env bash
# Verification suite:
#  1. Xqnn instruction tests (directed + random) on Spike and natively
#  2. Conv kernel tests for every kernel configuration on Spike
#     (+ accumulator-file sizes for the sparse custom kernel)
#  3. Full-model golden check: every config on Spike vs Python integer reference
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
SPIKE="tools/spike/bin/spike --isa=rv64imc_zicsr --extension=xqnn"
MODEL="${MODEL:-dense}"
fail=0
pass() { echo "  [PASS] $*"; }
bad()  { echo "  [FAIL] $*"; fail=1; }

echo "== 1. Xqnn instruction tests"
make -s build/tests/test_xqnn.elf build/tests/test_xqnn_native || exit 1
out=$($SPIKE build/tests/test_xqnn.elf); echo "$out" | sed 's/^/    /'
echo "$out" | grep -q "TOTAL .* -> PASS" && pass "spike" || bad "spike"
out=$(build/tests/test_xqnn_native); echo "$out" | grep -q -- "-> PASS" && pass "native emulation" || bad "native emulation"

echo "== 2. Conv kernel tests"
RV=tools/riscv/bin/riscv64-unknown-elf-gcc
CF="-march=rv64imc_zicsr -mabi=lp64 -mcmodel=medany -O2 -fno-schedule-insns -ffreestanding -nostdlib -fno-builtin -fno-tree-loop-distribute-patterns -Isrc/runtime -Isrc/nn -Wno-unused-function"
mkdir -p build/tests
for cfg in 0:B0 1:B1 2:X1 3:X2 4:S0 5:S1 6:S2 6:S2g2 6:S2g4 6:S2g16; do
  n=${cfg%%:*}; name=${cfg##*:}; g=8
  case $name in S2g2) g=2;; S2g4) g=4;; S2g16) g=16;; esac
  $RV $CF -DCFG=$n -DXQNN_NGRP=$g -T src/runtime/link.ld -Wl,--no-warn-rwx-segments \
     src/runtime/crt.S src/runtime/runtime.c src/tests/test_kernels.c -o build/tests/test_kernels_$name.elf || { bad "build $name"; continue; }
  out=$(XQNN_NGRP=$g $SPIKE build/tests/test_kernels_$name.elf)
  echo "$out" | grep -q "RESULT PASS" && pass "kernels $name" || { echo "$out"; bad "kernels $name"; }
done

echo "== 3. Full-model golden check ($MODEL model, 1 image, all configs)"
for c in B0 B1 X1 X2 S0 S1 S2; do
  make -s elf MODEL=$MODEL CFG=$c NIMG=1 >/dev/null || { bad "build $c"; continue; }
  out=$($SPIKE build/$MODEL/${c}_n1_g8_d1001/resnet18.elf | grep RESULT)
  echo "$out" | grep -q PASS && pass "$c: $out" || bad "$c: $out"
done

[ $fail = 0 ] && echo "ALL TESTS PASSED" || echo "SOME TESTS FAILED"
exit $fail
