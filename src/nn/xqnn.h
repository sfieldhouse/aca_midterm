// Xqnn custom instructions: inline-asm wrappers (RISC-V) or C emulation
// (native host build / -DXQNN_EMULATE).  Semantics: see src/spike_ext/xqnn.cc
// and DESIGN.md.  All use the custom-0 opcode, R-type, funct7 = 0.
#ifndef XQNN_H
#define XQNN_H
#include <stdint.h>

#ifndef XQNN_NGRP
#define XQNN_NGRP 8          // accumulator groups (8 accumulators each)
#endif

// requant parameter word for xrequant
#define XRQ_WORD(M, sh, zp, T, mode) \
  ((uint64_t)(uint32_t)(M) | ((uint64_t)((sh) & 63) << 32) | ((uint64_t)(uint8_t)(zp) << 40) | \
   ((uint64_t)(uint8_t)(T) << 48) | ((uint64_t)((mode) & 3) << 56))

#if defined(__riscv) && !defined(XQNN_EMULATE)

static inline int64_t xqmac8(int64_t acc, uint64_t a, uint64_t w)
{
  __asm__(".insn r CUSTOM_0, 0, 0, %0, %1, %2" : "+r"(acc) : "r"(a), "r"(w));
  return acc;
}
static inline int64_t xqmac8u(int64_t acc, uint64_t a, uint64_t w)
{
  __asm__(".insn r CUSTOM_0, 1, 0, %0, %1, %2" : "+r"(acc) : "r"(a), "r"(w));
  return acc;
}
static inline int64_t xrequant(int64_t acc, uint64_t prm)
{
  int64_t r;
  __asm__(".insn r CUSTOM_0, 2, 0, %0, %1, %2" : "=r"(r) : "r"(acc), "r"(prm));
  return r;
}
// G must be a literal integer (accumulator group, encoded in the rd field)
#define XSQMAC8(G, a, w) \
  __asm__ volatile(".insn r CUSTOM_0, 3, 0, x" #G ", %0, %1" ::"r"((uint64_t)(a)), "r"((uint64_t)(w)))
static inline int32_t xsaccrd(uint64_t idx)
{
  int64_t r;
  __asm__ volatile(".insn r CUSTOM_0, 4, 0, %0, %1, x0" : "=r"(r) : "r"(idx));
  return (int32_t)r;
}
static inline void xsaccclr(void) { __asm__ volatile(".insn r CUSTOM_0, 5, 0, x0, x0, x0"); }

#else  // ---------------------------------------------------------------- emulation

static inline int64_t xq_mac8_emu(int64_t acc, uint64_t a, uint64_t w, int ua)
{
  for (int i = 0; i < 8; i++) {
    int64_t ai = ua ? (int64_t)(uint8_t)(a >> (8 * i)) : (int64_t)(int8_t)(a >> (8 * i));
    acc += ai * (int8_t)(w >> (8 * i));
  }
  return acc;
}
static inline int64_t xqmac8(int64_t acc, uint64_t a, uint64_t w) { return xq_mac8_emu(acc, a, w, 0); }
static inline int64_t xqmac8u(int64_t acc, uint64_t a, uint64_t w) { return xq_mac8_emu(acc, a, w, 1); }
static inline int64_t xrequant(int64_t acc, uint64_t prm)
{
  uint64_t M = prm & 0xffffffffu;
  int sh = (prm >> 32) & 63;
  int64_t zp = (int8_t)(prm >> 40), T = (uint8_t)(prm >> 48);
  int mode = (prm >> 56) & 3;
  __int128 p = (__int128)acc * (__int128)M;
  if (sh) p = (p + ((__int128)1 << (sh - 1))) >> sh;
  p += zp;
  int64_t lo = mode == 1 ? 0 : -128, hi = mode == 1 ? 255 : 127;
  int64_t r = p < lo ? lo : p > hi ? hi : (int64_t)p;
  if (mode == 1 && r < T) r = 0;
  return r;
}
extern int32_t xq_acc_emu[32 * 8];
#define XSQMAC8(G, a, w)                                                         \
  do {                                                                           \
    int64_t a_ = (uint8_t)(a);                                                   \
    uint64_t w_ = (w);                                                           \
    for (int i_ = 0; i_ < 8; i_++)                                               \
      xq_acc_emu[(G) * 8 + i_] = (int32_t)((uint32_t)xq_acc_emu[(G) * 8 + i_] + \
                                           (uint32_t)(a_ * (int8_t)(w_ >> (8 * i_)))); \
  } while (0)
static inline int32_t xsaccrd(uint64_t idx)
{
  int32_t v = xq_acc_emu[idx];
  xq_acc_emu[idx] = 0;
  return v;
}
static inline void xsaccclr(void)
{
  for (int i = 0; i < 32 * 8; i++) xq_acc_emu[i] = 0;
}
#endif
#endif
