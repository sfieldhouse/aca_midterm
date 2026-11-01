// Xqnn: custom RISC-V extension for quantized / activation-sparse CNN inference.
//
// All instructions use the custom-0 major opcode (0b0001011 = 0x0B), R-type,
// funct7 = 0, selected by funct3:
//
//   funct3 0  xqmac8   rd, rs1, rs2   rd += sum_{i<8} s8(rs1.b[i]) * s8(rs2.b[i])
//   funct3 1  xqmac8u  rd, rs1, rs2   rd += sum_{i<8} u8(rs1.b[i]) * s8(rs2.b[i])
//   funct3 2  xrequant rd, rs1, rs2   rd = requant(acc = rs1, params = rs2)  (see below)
//   funct3 3  xsqmac8  g,  rs1, rs2   ACC[g][i] += u8(rs1[7:0]) * s8(rs2.b[i]), i<8
//                                     (rd field = accumulator group g, no GPR written)
//   funct3 4  xsaccrd  rd, rs1        rd = sext32(ACC[x[rs1]]); ACC[x[rs1]] = 0
//   funct3 5  xsaccclr                ACC[*] = 0
//
// xrequant parameter word rs2:
//   [31:0]  M     unsigned multiplier
//   [37:32] sh    right shift (0..63)
//   [47:40] zp    signed output zero point
//   [55:48] T     unsigned FATReLU threshold (only in mode 1)
//   [57:56] mode  0: saturate to int8 [-128,127]
//                 1: saturate to uint8 [0,255], then y < T -> 0 (ReLU / FATReLU)
//   y = sat( ((rs1 * M + (sh ? 1 << (sh-1) : 0)) >>arith sh) + zp )
//   (rs1 is a signed 64-bit accumulator; the product is computed exactly.)
//
// The accumulator file ACC holds NGRP*8 signed 32-bit accumulators (wrapping).
// NGRP defaults to 8 (64 accumulators) and can be changed with the environment
// variable XQNN_NGRP (1..32) for design-space exploration.  Using a group or
// index outside the file raises an illegal-instruction exception.

#include "insn_macros.h"
#include "extension.h"
#include "processor.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

static constexpr int XQNN_MAXGRP = 32;

static struct {
  int32_t acc[XQNN_MAXGRP * 8];
  int ngrp = 8;
} xq;

// Direct register-file access (decode_macros.h needs per-insn template context).
#define XRS1 (p->get_state()->XPR[insn.rs1()])
#define XRS2 (p->get_state()->XPR[insn.rs2()])
#define XRD_OLD (p->get_state()->XPR[insn.rd()])
static inline void write_rd(processor_t* p, insn_t insn, int64_t v)
{
  p->get_state()->XPR.write(insn.rd(), (reg_t)v);  // writes to x0 are discarded by XPR
}

static int xqnn_ngrp()
{
  const char* s = getenv("XQNN_NGRP");
  int n = s ? atoi(s) : 8;
  return (n >= 1 && n <= XQNN_MAXGRP) ? n : 8;
}

static reg_t xqmac8_common(processor_t* p, insn_t insn, reg_t pc, bool unsigned_a)
{
  uint64_t a = XRS1, w = XRS2;
  int64_t sum = 0;
  for (int i = 0; i < 8; i++) {
    int64_t ai = unsigned_a ? (int64_t)(uint8_t)(a >> (8 * i)) : (int64_t)(int8_t)(a >> (8 * i));
    int64_t wi = (int8_t)(w >> (8 * i));
    sum += ai * wi;
  }
  write_rd(p, insn, (int64_t)(XRD_OLD + (reg_t)sum));
  return pc + 4;
}

static reg_t insn_xqmac8(processor_t* p, insn_t insn, reg_t pc) { return xqmac8_common(p, insn, pc, false); }
static reg_t insn_xqmac8u(processor_t* p, insn_t insn, reg_t pc) { return xqmac8_common(p, insn, pc, true); }

static reg_t insn_xrequant(processor_t* p, insn_t insn, reg_t pc)
{
  int64_t acc = (int64_t)XRS1;
  uint64_t prm = XRS2;
  uint64_t M = prm & 0xffffffffu;
  int sh = (prm >> 32) & 0x3f;
  int64_t zp = (int8_t)(prm >> 40);
  int64_t T = (uint8_t)(prm >> 48);
  int mode = (prm >> 56) & 3;
  __int128 prod = (__int128)acc * (__int128)M;
  if (sh)
    prod = (prod + ((__int128)1 << (sh - 1))) >> sh;
  __int128 y = prod + zp;
  int64_t lo = mode == 1 ? 0 : -128, hi = mode == 1 ? 255 : 127;
  int64_t r = y < lo ? lo : y > hi ? hi : (int64_t)y;
  if (mode == 1 && r < T)
    r = 0;
  write_rd(p, insn, r);
  return pc + 4;
}

static reg_t insn_xsqmac8(processor_t* p, insn_t insn, reg_t pc)
{
  unsigned g = insn.rd();
  if ((int)g >= xq.ngrp)
    throw trap_illegal_instruction(insn.bits());
  int64_t a = (uint8_t)XRS1;
  uint64_t w = XRS2;
  int32_t* acc = &xq.acc[g * 8];
  for (int i = 0; i < 8; i++)
    acc[i] = (int32_t)((uint32_t)acc[i] + (uint32_t)(a * (int8_t)(w >> (8 * i))));
  return pc + 4;
}

static reg_t insn_xsaccrd(processor_t* p, insn_t insn, reg_t pc)
{
  reg_t idx = XRS1;
  if (idx >= (reg_t)xq.ngrp * 8)
    throw trap_illegal_instruction(insn.bits());
  int32_t v = xq.acc[idx];
  xq.acc[idx] = 0;
  write_rd(p, insn, (int64_t)v);
  return pc + 4;
}

static reg_t insn_xsaccclr(processor_t* p, insn_t insn, reg_t pc)
{
  memset(xq.acc, 0, sizeof(xq.acc));
  return pc + 4;
}

#define XQ_OP 0x0Bu
#define XQ_MASK 0xFE00707Fu
#define XQ_MATCH(f3) (XQ_OP | ((f3) << 12))
#define XQ_DESC(f3, fn) {XQ_MATCH(f3), XQ_MASK, fn, fn, fn, fn, fn, fn, fn, fn}

struct : public arg_t {
  std::string to_string(insn_t insn) const { return xpr_name[insn.rd()]; }
} xq_rd;
struct : public arg_t {
  std::string to_string(insn_t insn) const { return xpr_name[insn.rs1()]; }
} xq_rs1;
struct : public arg_t {
  std::string to_string(insn_t insn) const { return xpr_name[insn.rs2()]; }
} xq_rs2;
struct : public arg_t {
  std::string to_string(insn_t insn) const { return "acc" + std::to_string(insn.rd()); }
} xq_grp;

class xqnn_t : public extension_t
{
 public:
  const char* name() const override { return "xqnn"; }

  xqnn_t() { xq.ngrp = xqnn_ngrp(); }

  void reset(processor_t&) override { memset(xq.acc, 0, sizeof(xq.acc)); }

  std::vector<insn_desc_t> get_instructions(const processor_t&) override
  {
    return {
      XQ_DESC(0, insn_xqmac8),
      XQ_DESC(1, insn_xqmac8u),
      XQ_DESC(2, insn_xrequant),
      XQ_DESC(3, insn_xsqmac8),
      XQ_DESC(4, insn_xsaccrd),
      XQ_DESC(5, insn_xsaccclr),
    };
  }

  std::vector<disasm_insn_t*> get_disasms(const processor_t*) override
  {
    return {
      new disasm_insn_t("xqmac8", XQ_MATCH(0), XQ_MASK, {&xq_rd, &xq_rs1, &xq_rs2}),
      new disasm_insn_t("xqmac8u", XQ_MATCH(1), XQ_MASK, {&xq_rd, &xq_rs1, &xq_rs2}),
      new disasm_insn_t("xrequant", XQ_MATCH(2), XQ_MASK, {&xq_rd, &xq_rs1, &xq_rs2}),
      new disasm_insn_t("xsqmac8", XQ_MATCH(3), XQ_MASK, {&xq_grp, &xq_rs1, &xq_rs2}),
      new disasm_insn_t("xsaccrd", XQ_MATCH(4), XQ_MASK, {&xq_rd, &xq_rs1}),
      new disasm_insn_t("xsaccclr", XQ_MATCH(5), XQ_MASK, {}),
    };
  }
};

REGISTER_EXTENSION(xqnn, []() { static xqnn_t ext; return &ext; })
