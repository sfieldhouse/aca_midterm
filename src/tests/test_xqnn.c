// Directed + randomized verification of the Xqnn custom instructions against
// independent base-ISA C reference models.  Runs bare-metal on Spike
// (spike --isa=rv64imc_xqnn) and natively (emulation) for cross-checking.
#include <stdint.h>
#include "runtime.h"
#include "xqnn.h"

#ifndef NRAND
#define NRAND 20000
#endif

static uint64_t s = 0x9E3779B97F4A7C15ull;
static uint64_t rnd(void)
{  // xorshift64*
  s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
  return s * 0x2545F4914F6CDD1Dull;
}

static int fails, checks;
static void check(const char* what, uint64_t got, uint64_t exp, uint64_t a, uint64_t b, uint64_t c)
{
  checks++;
  if (got != exp) {
    if (fails < 20)
      printf("FAIL %s: got %lx exp %lx (ops %lx %lx %lx)\n", what, got, exp, a, b, c);
    fails++;
  }
}

// ---------------------------------------------------------------- references
static int64_t ref_qmac(int64_t acc, uint64_t a, uint64_t w, int ua)
{
  int64_t s = 0;
  for (int i = 0; i < 8; i++) {
    int64_t ai = (a >> (8 * i)) & 0xff, wi = (w >> (8 * i)) & 0xff;
    if (!ua && ai >= 128) ai -= 256;
    if (wi >= 128) wi -= 256;
    s += ai * wi;
  }
  return (int64_t)((uint64_t)acc + (uint64_t)s);
}

static int64_t ref_requant(int64_t acc, uint64_t prm)
{
  uint64_t M = (uint32_t)prm;
  unsigned sh = (prm >> 32) & 63;
  int64_t zp = (prm >> 40) & 0xff; if (zp >= 128) zp -= 256;
  int64_t T = (prm >> 48) & 0xff;
  unsigned mode = (prm >> 56) & 3;
  // exact 96-bit product via 32x32 partial products, sign-magnitude
  int neg = acc < 0;
  uint64_t mag = neg ? (uint64_t)0 - (uint64_t)acc : (uint64_t)acc;
  unsigned __int128 p = (unsigned __int128)(uint32_t)mag * M +
                        ((unsigned __int128)((mag >> 32) * M) << 32);
  __int128 v = neg ? -(__int128)p : (__int128)p;
  if (sh) {
    v += (__int128)1 << (sh - 1);
    // arithmetic shift right = floor division by 2^sh
    __int128 q = v >= 0 ? (v >> sh) : -(((-v) + (((__int128)1 << sh) - 1)) >> sh);
    v = q;
  }
  v += zp;
  int64_t lo = mode == 1 ? 0 : -128, hi = mode == 1 ? 255 : 127;
  int64_t r = v < lo ? lo : v > hi ? hi : (int64_t)v;
  if (mode == 1 && r < T) r = 0;
  return r;
}

// ---------------------------------------------------------------- xsqmac8 helpers
static int32_t shadow[XQNN_NGRP * 8];

static void ref_smac(int g, uint64_t a, uint64_t w)
{
  int64_t av = a & 0xff;
  for (int i = 0; i < 8; i++) {
    int64_t wi = (w >> (8 * i)) & 0xff;
    if (wi >= 128) wi -= 256;
    shadow[g * 8 + i] = (int32_t)((uint32_t)shadow[g * 8 + i] + (uint32_t)(av * wi));
  }
}

static void hw_smac(int g, uint64_t a, uint64_t w)
{
  switch (g) {  // group is an instruction immediate (rd field)
    case 0: XSQMAC8(0, a, w); break;
    case 1: XSQMAC8(1, a, w); break;
    case 2: XSQMAC8(2, a, w); break;
    case 3: XSQMAC8(3, a, w); break;
    case 4: XSQMAC8(4, a, w); break;
    case 5: XSQMAC8(5, a, w); break;
    case 6: XSQMAC8(6, a, w); break;
    case 7: XSQMAC8(7, a, w); break;
  }
}

int main(void)
{
  // ---------------- xqmac8 / xqmac8u directed
  const uint64_t pat[] = {0, 0x8080808080808080ull, 0x7f7f7f7f7f7f7f7full, 0xffffffffffffffffull,
                          0x0102030405060708ull, 0x80ff7f0001fe8081ull, 0x00000000000000ffull};
  const int64_t accs[] = {0, 1, -1, INT64_MAX, INT64_MIN, 123456789};
  for (int i = 0; i < 7; i++)
    for (int j = 0; j < 7; j++)
      for (int k = 0; k < 6; k++) {
        check("xqmac8", xqmac8(accs[k], pat[i], pat[j]), ref_qmac(accs[k], pat[i], pat[j], 0), accs[k], pat[i], pat[j]);
        check("xqmac8u", xqmac8u(accs[k], pat[i], pat[j]), ref_qmac(accs[k], pat[i], pat[j], 1), accs[k], pat[i], pat[j]);
      }
  // spot values
  check("xqmac8 -128*-128*8", xqmac8(0, pat[1], pat[1]), 131072, 0, 0, 0);
  check("xqmac8u 255*-128*8", xqmac8u(0, pat[3], pat[1]), (uint64_t)(int64_t)-261120, 0, 0, 0);
  // random
  for (int n = 0; n < NRAND; n++) {
    uint64_t a = rnd(), w = rnd();
    int64_t c = (int64_t)rnd() >> (rnd() & 63);
    check("xqmac8 rnd", xqmac8(c, a, w), ref_qmac(c, a, w, 0), c, a, w);
    check("xqmac8u rnd", xqmac8u(c, a, w), ref_qmac(c, a, w, 1), c, a, w);
  }
  printf("xqmac8/xqmac8u: %d checks, %d fails\n", checks, fails);
  int c0 = checks, f0 = fails;

  // ---------------- xrequant directed
  const int64_t racc[] = {0, 1, -1, 127, 128, -129, 255, 256, 1000000, -1000000, INT32_MAX, INT32_MIN,
                          INT64_MAX, INT64_MIN, (int64_t)1 << 40, -((int64_t)1 << 40)};
  const uint32_t Ms[] = {0, 1, 0x40000000u, 0x7fffffffu, 0xffffffffu, 0x5a5a5a5au};
  const unsigned shs[] = {0, 1, 2, 15, 16, 31, 32, 62, 63};
  const int zps[] = {0, 1, -1, 127, -128};
  const unsigned Ts[] = {0, 1, 7, 128, 255};
  for (unsigned ia = 0; ia < sizeof racc / 8; ia++)
    for (unsigned im = 0; im < sizeof Ms / 4; im++)
      for (unsigned is = 0; is < sizeof shs / 4; is++)
        for (unsigned iz = 0; iz < 5; iz++)
          for (unsigned it = 0; it < 5; it++)
            for (unsigned mode = 0; mode < 2; mode++) {
              uint64_t prm = XRQ_WORD(Ms[im], shs[is], zps[iz], Ts[it], mode);
              check("xrequant", xrequant(racc[ia], prm), ref_requant(racc[ia], prm), racc[ia], prm, 0);
            }
  // rounding half-up and FAT threshold edge cases
  check("xrq round 3/2", xrequant(3, XRQ_WORD(1, 1, 0, 0, 0)), 2, 0, 0, 0);
  check("xrq round -3/2", xrequant(-3, XRQ_WORD(1, 1, 0, 0, 0)), (uint64_t)-1, 0, 0, 0);
  check("xrq T keep", xrequant(10, XRQ_WORD(1, 0, 0, 10, 1)), 10, 0, 0, 0);
  check("xrq T zero", xrequant(9, XRQ_WORD(1, 0, 0, 10, 1)), 0, 0, 0, 0);
  check("xrq relu", xrequant(-5, XRQ_WORD(1, 0, 0, 0, 1)), 0, 0, 0, 0);
  check("xrq sat u8", xrequant(300, XRQ_WORD(1, 0, 0, 0, 1)), 255, 0, 0, 0);
  check("xrq sat s8", xrequant(-300, XRQ_WORD(1, 0, 0, 0, 0)), (uint64_t)-128, 0, 0, 0);
  for (int n = 0; n < NRAND; n++) {
    int64_t c = (int64_t)rnd() >> (rnd() & 63);
    uint64_t prm = rnd() & 0x03ffffffffffffffull;
    if (n & 1) prm = (prm & ~(0x3full << 32)) | ((uint64_t)(20 + rnd() % 44) << 32);  // useful shifts
    prm &= ~(3ull << 56);
    prm |= (uint64_t)(rnd() & 1) << 56;
    check("xrequant rnd", xrequant(c, prm), ref_requant(c, prm), c, prm, 0);
  }
  printf("xrequant: %d checks, %d fails\n", checks - c0, fails - f0);
  c0 = checks; f0 = fails;

  // ---------------- xsqmac8 / xsaccrd / xsaccclr
  xsaccclr();
  for (int i = 0; i < XQNN_NGRP * 8; i++) shadow[i] = 0;
  int ng = XQNN_NGRP < 8 ? XQNN_NGRP : 8;
  // directed: extreme values to every lane of every group, then read & clear
  for (int g = 0; g < ng; g++) {
    hw_smac(g, 0xff, 0x8080808080808080ull); ref_smac(g, 0xff, 0x8080808080808080ull);
    hw_smac(g, 0x80, 0x7f81ff0001807f80ull); ref_smac(g, 0x80, 0x7f81ff0001807f80ull);
    hw_smac(g, 0xabcdef01, 0x0102030405060708ull); ref_smac(g, 0xabcdef01, 0x0102030405060708ull);  // only low byte used
  }
  for (int i = 0; i < ng * 8; i++) {
    check("xsaccrd", (uint64_t)(int64_t)xsaccrd(i), (uint64_t)(int64_t)shadow[i], i, 0, 0);
    shadow[i] = 0;
    check("xsaccrd clears", (uint64_t)(int64_t)xsaccrd(i), 0, i, 0, 0);
  }
  // 32-bit wrap-around: 70000 * 255*127 > 2^31
  for (int n = 0; n < 70000; n++) { hw_smac(0, 255, 0x7f); ref_smac(0, 255, 0x7f); }
  check("xsqmac8 wrap", (uint64_t)(int64_t)xsaccrd(0), (uint64_t)(int64_t)shadow[0], 0, 0, 0);
  shadow[0] = 0;
  for (int i = 1; i < 8; i++) { (void)xsaccrd(i); shadow[i] = 0; }
  // random sequences
  for (int n = 0; n < NRAND; n++) {
    int g = rnd() % ng;
    uint64_t a = rnd(), w = rnd();
    hw_smac(g, a, w); ref_smac(g, a, w);
    if ((rnd() & 15) == 0) {
      int i = rnd() % (ng * 8);
      check("xsaccrd rnd", (uint64_t)(int64_t)xsaccrd(i), (uint64_t)(int64_t)shadow[i], i, a, w);
      shadow[i] = 0;
    }
  }
  for (int i = 0; i < ng * 8; i++) {
    check("xsaccrd final", (uint64_t)(int64_t)xsaccrd(i), (uint64_t)(int64_t)shadow[i], i, 0, 0);
    shadow[i] = 0;
  }
  // clear-all
  for (int g = 0; g < ng; g++) hw_smac(g, 3, 0x0101010101010101ull);
  xsaccclr();
  for (int i = 0; i < ng * 8; i++) check("xsaccclr", (uint64_t)(int64_t)xsaccrd(i), 0, i, 0, 0);
  printf("xsqmac8/xsaccrd/xsaccclr: %d checks, %d fails\n", checks - c0, fails - f0);

  printf("TOTAL %d checks, %d fails -> %s\n", checks, fails, fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}

#if !defined(__riscv) || defined(XQNN_EMULATE)
int32_t xq_acc_emu[32 * 8];
#endif
