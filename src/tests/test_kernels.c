// Kernel-level verification: the conv kernels of the selected CFG (dense
// blocked / sparse CSR, scalar or custom instructions) against an independent
// reference convolution + requantisation, on random tensors with varying
// shapes, strides, densities and requant modes.  Also checks csr_compress.
#include <stdint.h>
#include "nn.h"
#include "runtime.h"
#include "xqnn.h"

#include "../nn/kernels.c"

const conv_desc_t g_convs[20];
const add_desc_t g_adds[8];
const pool_desc_t g_pool;
const fc_desc_t g_fc;
const uint8_t* g_blob;

#define MAXB (1 << 20)
static uint8_t blobbuf[MAXB] __attribute__((aligned(64)));
static uint8_t inbuf[18 * 18 * 128] __attribute__((aligned(64)));
static uint8_t outk[18 * 18 * 128] __attribute__((aligned(64)));
static uint8_t outr[18 * 18 * 128] __attribute__((aligned(64)));
static int8_t wref[128 * 128 * 9];
static uint32_t ent[16 * 16 * 128], rowp[18 * 18 + 1];

static uint64_t s = 88172645463325252ull;
static uint32_t rnd(void)
{
  s ^= s << 13; s ^= s >> 7; s ^= s << 17;
  return (uint32_t)(s >> 16);
}

static uint32_t boff;
static uint32_t balloc(uint32_t n)
{
  boff = (boff + 63) & ~63u;
  uint32_t o = boff;
  boff += n;
  return o;
}

static int run_case(int ic, int oc, int h, int k, int stride, int pad, int dens_pm, int mode, int T)
{
  int oh = (h + 2 * pad - k) / stride + 1, pw = h + 2, opw = oh + 2;
  conv_desc_t d = {"test", ic, oc, h, h, oh, oh, k, stride, pad, k, k * ic, pw, 1 - pad, mode, T, 0, 0, 0, 0, 0, 0, 0, 0};
  boff = 0;
  // random weights W[oc][ic][kh][kw], bias, requant params
  for (int i = 0; i < oc * ic * k * k; i++) wref[i] = (int8_t)(rnd() % 255 - 127);
  d.w_naive = balloc(oc * k * k * ic);
  d.w_blk = balloc(oc * k * k * ic);
  d.w_sp = balloc(oc * k * k * ic);
  d.bias = balloc(4 * oc); d.mult = balloc(4 * oc); d.shift = balloc(4 * oc); d.rqw = balloc(8 * oc);
  int8_t *wn = (int8_t*)(blobbuf + d.w_naive), *wb = (int8_t*)(blobbuf + d.w_blk), *ws = (int8_t*)(blobbuf + d.w_sp);
  int L = k * ic;
  for (int o = 0; o < oc; o++)
    for (int kh = 0; kh < k; kh++)
      for (int kw = 0; kw < k; kw++)
        for (int c = 0; c < ic; c++) {
          int8_t w = wref[((o * ic + c) * k + kh) * k + kw];
          int j = kw * ic + c;
          wn[(o * k + kh) * L + j] = w;
          wb[((((o / 4) * k + kh) * (L / 8) + j / 8) * 4 + o % 4) * 8 + j % 8] = w;
          ws[((kh * k + kw) * ic + c) * oc + o] = w;
        }
  int32_t *bias = (int32_t*)(blobbuf + d.bias), *M = (int32_t*)(blobbuf + d.mult), *S = (int32_t*)(blobbuf + d.shift);
  uint64_t* W = (uint64_t*)(blobbuf + d.rqw);
  for (int o = 0; o < oc; o++) {
    bias[o] = (int32_t)(rnd() % 20001) - 10000;
    M[o] = (int32_t)(0x40000000u + rnd() % 0x3fffffffu);
    S[o] = 38 + rnd() % 6;
    W[o] = XRQ_WORD(M[o], S[o], 0, T, mode);
  }
  g_blob = blobbuf;
  // random bordered input with given density
  for (int i = 0; i < pw * pw * ic; i++) inbuf[i] = 0;
  for (int y = 1; y <= h; y++)
    for (int x = 1; x <= h; x++)
      for (int c = 0; c < ic; c++)
        if ((int)(rnd() % 1000) < dens_pm) inbuf[(y * pw + x) * ic + c] = 1 + rnd() % 255;
  // reference
  for (int i = 0; i < opw * opw * oc; i++) { outr[i] = 0; outk[i] = 0xAA; }
  zero_border(outk, oh, oh, oc);
  for (int oy = 0; oy < oh; oy++)
    for (int ox = 0; ox < oh; ox++)
      for (int o = 0; o < oc; o++) {
        int64_t acc = bias[o];
        for (int kh = 0; kh < k; kh++)
          for (int kw = 0; kw < k; kw++) {
            int iy = oy * stride - pad + kh, ix = ox * stride - pad + kw;
            if (iy < 0 || ix < 0 || iy >= h || ix >= h) continue;
            for (int c = 0; c < ic; c++)
              acc += (int64_t)inbuf[((iy + 1) * pw + ix + 1) * ic + c] * wref[((o * ic + c) * k + kh) * k + kw];
          }
        int64_t v = (acc * M[o] + ((int64_t)1 << (S[o] - 1))) >> S[o];
        int lo = mode ? 0 : -128, hi = mode ? 255 : 127;
        v = v < lo ? lo : v > hi ? hi : v;
        if (mode && v < T) v = 0;
        outr[((oy + 1) * opw + ox + 1) * oc + o] = (uint8_t)v;
      }
  // kernel under test
#if CFG_NAIVE
  conv_naive(&d, inbuf, outk);
#elif CFG_SPARSE
  uint32_t nnz = csr_compress(inbuf, h, h, ic, oc, ent, rowp);
  uint32_t cnt = 0;
  for (int i = 0; i < pw * pw * ic; i++) cnt += inbuf[i] != 0;
  if (nnz != cnt || rowp[pw * pw] != nnz) { printf("  csr nnz mismatch %u %u\n", nnz, cnt); return 1; }
  conv_sparse(&d, ent, rowp, outk);
#else
  conv_dense(&d, inbuf, outk);
#endif
  int bad = 0;
  for (int i = 0; i < opw * opw * oc; i++) bad += outk[i] != outr[i];
  printf("  ic=%3d oc=%3d h=%2d k=%d s=%d p=%d density=%4d/1000 mode=%d T=%2d : %s (%d bad)\n",
         ic, oc, h, k, stride, pad, dens_pm, mode, T, bad ? "FAIL" : "ok", bad);
  return bad != 0;
}

int main(void)
{
  printf("kernel tests CFG=%d NGRP=%d\n", CFG, XQNN_NGRP);
  int f = 0;
  f += run_case(64, 64, 8, 3, 1, 1, 300, 1, 5);
  f += run_case(64, 64, 7, 3, 1, 1, 1000, 1, 0);     // odd pixel count, fully dense
  f += run_case(64, 128, 8, 3, 2, 1, 500, 1, 3);     // stride 2
  f += run_case(64, 128, 8, 1, 2, 0, 400, 0, 0);     // 1x1 downsample, int8 output
  f += run_case(128, 128, 4, 3, 1, 1, 50, 0, 0);     // very sparse
  f += run_case(128, 64, 6, 3, 1, 1, 0, 1, 0);       // all-zero input
  f += run_case(128, 128, 5, 3, 1, 1, 700, 1, 20);   // high FATReLU threshold
  printf("RESULT %s\n", f ? "FAIL" : "PASS");
  return f;
}

#if !defined(__riscv) || defined(XQNN_EMULATE)
int32_t xq_acc_emu[32 * 8];
#endif
