#!/usr/bin/env bash
# Build Spike (third_party/riscv-isa-sim, pinned) with the Xqnn custom extension
# (src/spike_ext/xqnn.cc) linked statically into the spike binary, and install
# it into tools/spike.  Use:  spike --isa=rv64imc_zicsr --extension=xqnn prog.elf
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SPIKE_SRC="$ROOT/third_party/riscv-isa-sim"
PREFIX="$ROOT/tools/spike"
DTC_DIR="${DTC_DIR:-$(dirname "$(command -v dtc || echo "$HOME/programs/dtc-1.7.0/dtc")")}"
export PATH="$DTC_DIR:$PATH"
command -v dtc >/dev/null || { echo "dtc (device-tree-compiler) not found; set DTC_DIR"; exit 1; }

# Hook the extension into libriscv and force-link its object into spike (idempotent).
# (Spike's dlopen()ed libcustomext.so re-registers built-in plugins with a
#  statically linked spike, so the extension is compiled in instead.)
ln -sf "$ROOT/src/spike_ext/xqnn.cc" "$SPIKE_SRC/riscv/xqnn.cc"
grep -q "xqnn.cc" "$SPIKE_SRC/riscv/riscv.mk.in" || \
  sed -i 's/^riscv_srcs = \\$/riscv_srcs = \\\n\txqnn.cc \\/' "$SPIKE_SRC/riscv/riscv.mk.in"
grep -q "xqnn.o" "$SPIKE_SRC/spike_main/spike_main.mk.in" || \
  sed -i 's/^spike_main_LDFLAGS = extension.o$/spike_main_LDFLAGS = extension.o xqnn.o/' "$SPIKE_SRC/spike_main/spike_main.mk.in"
grep -q "xqnn.cc" "$SPIKE_SRC/riscv/riscv.mk.in"
grep -q "xqnn.o" "$SPIKE_SRC/spike_main/spike_main.mk.in"

mkdir -p "$SPIKE_SRC/build"
cd "$SPIKE_SRC/build"
[ -f config.status ] || ../configure --prefix="$PREFIX" --without-boost --without-boost-asio --without-boost-regex
./config.status >/dev/null
make -j"$(nproc)"
make install
echo "spike installed in $PREFIX/bin/spike (use --isa=rv64imc_zicsr --extension=xqnn)"
