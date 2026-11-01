#!/usr/bin/env bash
# Download a prebuilt RISC-V bare-metal GCC toolchain (no sudo needed) into tools/riscv.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG="${TOOLCHAIN_TAG:-2026.08.27}"
URL="https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/${TAG}/riscv64-elf-ubuntu-22.04-gcc.tar.xz"
DEST="$ROOT/tools"
mkdir -p "$DEST"
if [ -x "$DEST/riscv/bin/riscv64-unknown-elf-gcc" ]; then
  echo "toolchain already installed at $DEST/riscv"; exit 0
fi
echo "Downloading $URL"
curl -L --fail -o "$DEST/toolchain.tar.xz" "$URL"
tar -C "$DEST" -xf "$DEST/toolchain.tar.xz"
rm -f "$DEST/toolchain.tar.xz"
"$DEST/riscv/bin/riscv64-unknown-elf-gcc" --version | head -1
