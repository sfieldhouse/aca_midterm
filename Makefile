# Build / run the bare-metal ResNet-18 workload and the Xqnn tests.
#
#   make setup                       toolchain + spike (first time)
#   make test                        custom-instruction verification (Spike + native)
#   make elf   MODEL=sparse CFG=S2 NIMG=4 [NGRP=8] [DISPATCH=1001]
#   make run   MODEL=sparse CFG=S2 NIMG=4      build + run on Spike
#   make native MODEL=sparse CFG=S2            host build (emulated custom insns)
#
# CFG: B0 naive | B1 optimized | X1 +xqmac8 | X2 +xqmac8+xrequant |
#      S0 sparse SW | S1 sparse+xsqmac8 | S2 sparse+xsqmac8+xqmac8+xrequant
ROOT    := $(CURDIR)
RISCV   ?= $(ROOT)/tools/riscv/bin/riscv64-unknown-elf-
SPIKE   ?= $(ROOT)/tools/spike/bin/spike
CC      := $(RISCV)gcc
OBJDUMP := $(RISCV)objdump
SIZE    := $(RISCV)size
ARCH    := -march=rv64imc_zicsr -mabi=lp64 -mcmodel=medany
CFLAGS  := $(ARCH) -O2 -fno-schedule-insns -ffreestanding -nostdlib -fno-builtin -fno-tree-loop-distribute-patterns \
           -Wall -Wno-unused-function -Isrc/runtime -Isrc/nn
SPIKE_FLAGS := --isa=rv64imc_zicsr --extension=xqnn

MODEL    ?= sparse
CFG      ?= S2
NIMG     ?= 1
NGRP     ?= 8
DISPATCH ?= 1001
CFGNUM_B0 := 0
CFGNUM_B1 := 1
CFGNUM_X1 := 2
CFGNUM_X2 := 3
CFGNUM_S0 := 4
CFGNUM_S1 := 5
CFGNUM_S2 := 6
DEFS    := -DCFG=$(CFGNUM_$(CFG)) -DNIMG=$(NIMG) -DXQNN_NGRP=$(NGRP) -DSPARSE_DISPATCH_PERMILLE=$(DISPATCH)

EXPORT  := models/$(MODEL)/export
TAG     := $(CFG)_n$(NIMG)_g$(NGRP)_d$(DISPATCH)
OUT     := build/$(MODEL)/$(TAG)
ELF     := $(OUT)/resnet18.elf
NN_SRC  := src/nn/kernels.c src/nn/model.c src/nn/main.c
RT_SRC  := src/runtime/crt.S src/runtime/runtime.c
HDRS    := $(wildcard src/nn/*.h src/runtime/*.h)

.PHONY: setup test elf run native clean
setup:
	scripts/setup_toolchain.sh
	scripts/build_spike.sh

elf: $(ELF)
$(ELF): Makefile $(NN_SRC) $(RT_SRC) $(HDRS) $(EXPORT)/model_desc.c $(EXPORT)/inputs.c $(EXPORT)/blob.S $(EXPORT)/model.bin
	@mkdir -p $(OUT)
	$(CC) $(CFLAGS) $(DEFS) -T src/runtime/link.ld -Wl,--no-warn-rwx-segments \
	  $(RT_SRC) $(NN_SRC) $(EXPORT)/model_desc.c $(EXPORT)/inputs.c $(EXPORT)/blob.S -o $@
	$(OBJDUMP) -d $@ > $(OUT)/resnet18.dis
	$(SIZE) -A $@ > $(OUT)/size.txt

run: $(ELF)
	$(SPIKE) $(SPIKE_FLAGS) $(ELF)

native: build/native/$(MODEL)/$(TAG)/resnet18
build/native/$(MODEL)/$(TAG)/resnet18: $(NN_SRC) $(HDRS) src/runtime/native.c $(EXPORT)/model_desc.c $(EXPORT)/inputs.c
	@mkdir -p $(dir $@)
	gcc -O2 -DNATIVE $(DEFS) -Isrc/runtime -Isrc/nn $(NN_SRC) src/runtime/native.c \
	  $(EXPORT)/model_desc.c $(EXPORT)/inputs.c -o $@

# ---------------------------------------------------------------- tests
build/tests/test_xqnn.elf: src/tests/test_xqnn.c $(RT_SRC) $(HDRS)
	@mkdir -p build/tests
	$(CC) $(CFLAGS) -T src/runtime/link.ld -Wl,--no-warn-rwx-segments $(RT_SRC) $< -o $@
build/tests/test_xqnn_native: src/tests/test_xqnn.c src/runtime/native.c $(HDRS)
	@mkdir -p build/tests
	gcc -O2 -Isrc/runtime -Isrc/nn $< src/runtime/native.c -o $@
test: build/tests/test_xqnn.elf build/tests/test_xqnn_native
	MODEL=$(MODEL) scripts/run_tests.sh

clean:
	rm -rf build
