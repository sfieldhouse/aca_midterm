// Quantized ResNet-18 kernels for RV64IMC (+ optional Xqnn custom instructions).
// Kernel variant is selected at compile time by CFG (see nn.h).
#include "nn.h"
#include "xqnn.h"

// ======================================================================== requant
// Software requantisation (bit-identical to xrequant): lo/hi/T are loop
// invariants (mode 0: [-128,127], T = -1000; mode 1: [0,255], FATReLU T).
static inline int32_t rq_sw(int64_t acc, int32_t M, int32_t sh, int lo, int hi, int T)
{
  int64_t v = (acc * M + ((int64_t)1 << (sh - 1))) >> sh;
  if (v < lo) v = lo;
  if (v > hi) v = hi;
  if (v < T) v = 0;
  return (int32_t)v;
}

typedef struct {
  const int32_t *bias, *M, *S;
  const uint64_t* W;
  int lo, hi, T;
} rq_t;

static inline rq_t rq_init(const conv_desc_t* d)
{
  rq_t r;
  r.bias = blob(d->bias);
  r.M = blob(d->mult);
  r.S = blob(d->shift);
  r.W = blob(d->rqw);
  r.lo = d->mode ? 0 : -128;
  r.hi = d->mode ? 255 : 127;
  r.T = d->mode ? d->T : -1000;
  return r;
}

#if USE_XRQ
#define RQ(r, c, acc) ((int32_t)xrequant((acc), (r).W[c]))
#else
#define RQ(r, c, acc) rq_sw((acc), (r).M[c], (r).S[c], (r).lo, (r).hi, (r).T)
#endif

// ======================================================================== helpers
void zero_border(void* buf, int h, int w, int c)
{
  uint64_t* p = buf;
  int rowq = (w + 2) * c / 8, pixq = c / 8;
  for (int i = 0; i < rowq; i++) p[i] = 0;
  uint64_t* last = p + (h + 1) * rowq;
  for (int i = 0; i < rowq; i++) last[i] = 0;
  for (int y = 1; y <= h; y++) {
    uint64_t* r = p + y * rowq;
    for (int i = 0; i < pixq; i++) {
      r[i] = 0;
      r[(w + 1) * pixq + i] = 0;
    }
  }
}

static inline uint8_t* out_px(const conv_desc_t* d, void* out, int oy, int ox)
{
  return (uint8_t*)out + ((oy + 1) * (d->ow + 2) + ox + 1) * d->oc;
}

static inline int in_px(const conv_desc_t* d, int oy, int ox)
{
  return (oy * d->stride + d->in_off) * d->in_rowpix + ox * d->stride + d->in_off;
}

// ======================================================================== B0: naive dense
#define NAIVE_BODY(T_IN)                                                      \
  for (int oy = 0; oy < d->oh; oy++)                                          \
    for (int ox = 0; ox < d->ow; ox++) {                                      \
      const T_IN* base = (const T_IN*)in + in_px(d, oy, ox) * d->ic;          \
      uint8_t* o = out_px(d, out, oy, ox);                                    \
      for (int c = 0; c < d->oc; c++) {                                       \
        int64_t acc = r.bias[c];                                              \
        const int8_t* w = W + c * d->k * d->L;                                \
        for (int kh = 0; kh < d->k; kh++) {                                   \
          const T_IN* a = base + kh * d->in_rowpix * d->ic;                   \
          for (int j = 0; j < d->L; j++) acc += a[j] * w[j];                  \
          w += d->L;                                                          \
        }                                                                     \
        o[c] = (uint8_t)RQ(r, c, acc);                                        \
      }                                                                       \
    }

void conv_naive(const conv_desc_t* d, const void* in, void* out)
{
  const int8_t* W = blob(d->w_naive);
  rq_t r = rq_init(d);
  if (d->in_signed) {
    NAIVE_BODY(int8_t)
  } else {
    NAIVE_BODY(uint8_t)
  }
}

// ======================================================================== B1/X1/X2: blocked dense
// Scalar (B1): 2 output pixels x 4 output channels register block.
// Weights [oc/4][k][L/8][4][8].
#if !USE_XQMAC
#define DENSE_INNER(T_IN, MAC)                                                \
  for (int j = 0; j < L8; j++) {                                              \
    const T_IN* xa = (const T_IN*)(x0p + j);                                  \
    const T_IN* xb = (const T_IN*)(x1p + j);                                  \
    const int8_t* wb = (const int8_t*)wp;                                     \
    _Pragma("GCC unroll 8") for (int b = 0; b < 8; b++) {                     \
      int32_t p0 = xa[b], p1 = xb[b];                                         \
      int32_t w0 = wb[b], w1 = wb[8 + b], w2 = wb[16 + b], w3 = wb[24 + b];   \
      a00 += p0 * w0; a01 += p0 * w1; a02 += p0 * w2; a03 += p0 * w3;        \
      a10 += p1 * w0; a11 += p1 * w1; a12 += p1 * w2; a13 += p1 * w3;        \
    }                                                                         \
    wp += 4;                                                                  \
  }
#endif

#define DENSE_BODY(T_IN, MAC)                                                       \
  for (int p = 0; p < P; p += 2) {                                                  \
    int q = p + 1 < P ? p + 1 : p;                                                  \
    int oy0 = p / d->ow, ox0 = p - oy0 * d->ow, oy1 = q / d->ow, ox1 = q - oy1 * d->ow; \
    const uint8_t* b0 = (const uint8_t*)in + in_px(d, oy0, ox0) * d->ic;            \
    const uint8_t* b1 = (const uint8_t*)in + in_px(d, oy1, ox1) * d->ic;            \
    uint8_t* o0 = out_px(d, out, oy0, ox0);                                         \
    uint8_t* o1 = out_px(d, out, oy1, ox1);                                         \
    const uint64_t* wp = Wb;                                                        \
    for (int c = 0; c < d->oc; c += 4) {                                            \
      int64_t a00 = r.bias[c], a01 = r.bias[c + 1], a02 = r.bias[c + 2], a03 = r.bias[c + 3]; \
      int64_t a10 = a00, a11 = a01, a12 = a02, a13 = a03;                           \
      for (int kh = 0; kh < d->k; kh++) {                                           \
        const uint64_t* x0p = (const uint64_t*)(b0 + kh * rowb);                    \
        const uint64_t* x1p = (const uint64_t*)(b1 + kh * rowb);                    \
        DENSE_INNER(T_IN, MAC)                                                      \
      }                                                                             \
      o0[c] = RQ(r, c, a00); o0[c + 1] = RQ(r, c + 1, a01);                         \
      o0[c + 2] = RQ(r, c + 2, a02); o0[c + 3] = RQ(r, c + 3, a03);                 \
      if (q != p) {                                                                 \
        o1[c] = RQ(r, c, a10); o1[c + 1] = RQ(r, c + 1, a11);                       \
        o1[c + 2] = RQ(r, c + 2, a12); o1[c + 3] = RQ(r, c + 3, a13);               \
      }                                                                             \
    }                                                                               \
  }

#if USE_XQMAC
// 4 output pixels x 4 output channels = 16 accumulators (xqmac8 relieves the
// register pressure of the scalar version, so a larger block fits).
#define XM(i, j, MAC) a##i##j = MAC(a##i##j, x##i, w)
#define XROW(j, MAC) w = wp[j]; XM(0, j, MAC); XM(1, j, MAC); XM(2, j, MAC); XM(3, j, MAC)
#define XST(i)                                                                  \
  if (n > i) {                                                                  \
    o##i[c] = RQ(r, c, a##i##0); o##i[c + 1] = RQ(r, c + 1, a##i##1);           \
    o##i[c + 2] = RQ(r, c + 2, a##i##2); o##i[c + 3] = RQ(r, c + 3, a##i##3);   \
  }
#define XPIX(i)                                                                 \
  int q##i = p + i < P ? p + i : P - 1;                                         \
  int oy##i = q##i / d->ow, ox##i = q##i - oy##i * d->ow;                       \
  const uint8_t* b##i = (const uint8_t*)in + in_px(d, oy##i, ox##i) * d->ic;    \
  uint8_t* o##i = out_px(d, out, oy##i, ox##i);
#define DENSE4_BODY(MAC)                                                        \
  for (int p = 0; p < P; p += 4) {                                              \
    int n = P - p;                                                              \
    XPIX(0) XPIX(1) XPIX(2) XPIX(3)                                             \
    const uint64_t* wp = Wb;                                                    \
    for (int c = 0; c < d->oc; c += 4) {                                        \
      int64_t a00 = r.bias[c], a01 = r.bias[c + 1], a02 = r.bias[c + 2], a03 = r.bias[c + 3]; \
      int64_t a10 = a00, a11 = a01, a12 = a02, a13 = a03;                       \
      int64_t a20 = a00, a21 = a01, a22 = a02, a23 = a03;                       \
      int64_t a30 = a00, a31 = a01, a32 = a02, a33 = a03;                       \
      for (int kh = 0; kh < d->k; kh++) {                                       \
        const uint64_t* x0p = (const uint64_t*)(b0 + kh * rowb);                \
        const uint64_t* x1p = (const uint64_t*)(b1 + kh * rowb);                \
        const uint64_t* x2p = (const uint64_t*)(b2 + kh * rowb);                \
        const uint64_t* x3p = (const uint64_t*)(b3 + kh * rowb);                \
        for (int j = 0; j < L8; j++) {                                          \
          uint64_t x0 = x0p[j], x1 = x1p[j], x2 = x2p[j], x3 = x3p[j], w;       \
          XROW(0, MAC); XROW(1, MAC); XROW(2, MAC); XROW(3, MAC);                          \
          wp += 4;                                                              \
        }                                                                       \
      }                                                                         \
      XST(0) XST(1) XST(2) XST(3)                                               \
    }                                                                           \
  }

void conv_dense(const conv_desc_t* d, const void* in, void* out)
{
  const uint64_t* Wb = blob(d->w_blk);
  rq_t r = rq_init(d);
  int P = d->oh * d->ow, L8 = d->L / 8, rowb = d->in_rowpix * d->ic;
  if (d->in_signed) {
    // stem: kernel row = 8 px x 4 ch = 32 B = 4 chunks; a constant trip count lets
    // the compiler fully unroll the chunk loop
    if (L8 == 4) {
      const int L8 = 4;
      DENSE4_BODY(xqmac8)
    } else {
      DENSE4_BODY(xqmac8)
    }
  } else {
    DENSE4_BODY(xqmac8u)
  }
}
#else
void conv_dense(const conv_desc_t* d, const void* in, void* out)
{
  const uint64_t* Wb = blob(d->w_blk);
  rq_t r = rq_init(d);
  int P = d->oh * d->ow, L8 = d->L / 8, rowb = d->in_rowpix * d->ic;
  if (d->in_signed) {
    DENSE_BODY(int8_t, xqmac8)
  } else {
    DENSE_BODY(uint8_t, xqmac8u)
  }
}
#endif

// ======================================================================== CSR compression
// Input: bordered uint8 tensor (h+2)x(w+2)xc.  Row q = bordered pixel index.
// Entry = (ic * oc) << 8 | value   (byte offset of the weight row, activation).
uint32_t csr_compress(const uint8_t* in, int h, int w, int c, int oc, uint32_t* ent, uint32_t* rowptr)
{
  uint32_t n = 0;
  int pw = w + 2, cq = c / 8;
  uint32_t* rp = rowptr;
  for (int x = 0; x < pw; x++) *rp++ = 0;           // top border row: empty
  for (int y = 1; y <= h; y++) {
    *rp++ = n;                                      // left border pixel
    const uint64_t* p = (const uint64_t*)(in + (y * pw + 1) * c);
    for (int x = 1; x <= w; x++) {
      *rp++ = n;
      uint32_t coff = 0;
      for (int j = 0; j < cq; j++, coff += 8 * oc) {
        uint64_t v = *p++;
        uint32_t e = coff << 8;
        while (v) {
          uint32_t b = v & 0xff;
          if (b) ent[n++] = e | b;
          v >>= 8;
          e += (uint32_t)oc << 8;
        }
      }
    }
    *rp++ = n;                                      // right border pixel
  }
  for (int x = 0; x <= pw; x++) *rp++ = n;          // bottom border row + end pointer
  return n;
}

// ======================================================================== S0/S1/S2: sparse-input conv
// Paper Alg. 1/2: for every output pixel and output-channel block, walk the
// nonzero activations of the k*k input pixels and broadcast-MAC them into the
// block's accumulators.  Weights [kh][kw][ic][oc].
#if USE_XSMAC
#define SM(g) XSQMAC8(g, e, wq[g])
#define SM2 SM(0); SM(1)
#define SM4 SM2; SM(2); SM(3)
#define SM8 SM4; SM(4); SM(5); SM(6); SM(7)
#define SM16 SM8; SM(8); SM(9); SM(10); SM(11); SM(12); SM(13); SM(14); SM(15)

#define SPARSE_X(NAME, G, BODY)                                                       \
  static void NAME(const conv_desc_t* d, const uint32_t* ent, const uint32_t* rowptr, \
                   void* out, rq_t r)                                                 \
  {                                                                                   \
    const int8_t* W = blob(d->w_sp);                                                  \
    int tapb = d->ic * d->oc, rp = d->in_rowpix;                                      \
    for (int oy = 0; oy < d->oh; oy++)                                                \
      for (int ox = 0; ox < d->ow; ox++) {                                            \
        int qb = in_px(d, oy, ox);                                                    \
        uint8_t* o = out_px(d, out, oy, ox);                                          \
        for (int ob = 0; ob < d->oc; ob += 8 * (G)) {                                 \
          const int8_t* wt = W + ob;                                                  \
          for (int kh = 0; kh < d->k; kh++) {                                         \
            const uint32_t* rq = rowptr + qb + kh * rp;                               \
            for (int kw = 0; kw < d->k; kw++, wt += tapb) {                           \
              const uint32_t* ep = ent + rq[kw];                                      \
              const uint32_t* ee = ent + rq[kw + 1];                                  \
              for (; ep < ee; ep++) {                                                 \
                uint64_t e = *ep;  /* lwu: no extra zero-extension */                \
                const uint64_t* wq = (const uint64_t*)(wt + (e >> 8));                \
                BODY;                                                                 \
              }                                                                       \
            }                                                                         \
          }                                                                           \
          for (int i = 0; i < 8 * (G); i++)                                           \
            o[ob + i] = RQ(r, ob + i, (int64_t)xsaccrd(i) + r.bias[ob + i]);          \
        }                                                                             \
      }                                                                               \
  }

SPARSE_X(sparse_g2, 2, SM2)
SPARSE_X(sparse_g4, 4, SM4)
SPARSE_X(sparse_g8, 8, SM8)
#if XQNN_NGRP >= 16
SPARSE_X(sparse_g16, 16, SM16)
#endif

void conv_sparse(const conv_desc_t* d, const uint32_t* ent, const uint32_t* rowptr, void* out)
{
  rq_t r = rq_init(d);
  int g = d->oc / 8 < XQNN_NGRP ? d->oc / 8 : XQNN_NGRP;
#if XQNN_NGRP >= 16
  if (g >= 16) { sparse_g16(d, ent, rowptr, out, r); return; }
#endif
  if (g >= 8) sparse_g8(d, ent, rowptr, out, r);
  else if (g >= 4) sparse_g4(d, ent, rowptr, out, r);
  else sparse_g2(d, ent, rowptr, out, r);
}

#else  // ---------------------------------------------------------- S0: scalar, 16 GPR accumulators
#define SA(i) a##i += v * w[i]
#define SO(i) o[ob + i] = RQ(r, ob + i, (int64_t)a##i + r.bias[ob + i])
void conv_sparse(const conv_desc_t* d, const uint32_t* ent, const uint32_t* rowptr, void* out)
{
  rq_t r = rq_init(d);
  const int8_t* W = blob(d->w_sp);
  int tapb = d->ic * d->oc, rp = d->in_rowpix;
  for (int oy = 0; oy < d->oh; oy++)
    for (int ox = 0; ox < d->ow; ox++) {
      int qb = in_px(d, oy, ox);
      uint8_t* o = out_px(d, out, oy, ox);
      for (int ob = 0; ob < d->oc; ob += 16) {
        int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
        int32_t a8 = 0, a9 = 0, a10 = 0, a11 = 0, a12 = 0, a13 = 0, a14 = 0, a15 = 0;
        const int8_t* wt = W + ob;
        for (int kh = 0; kh < d->k; kh++) {
          const uint32_t* rq = rowptr + qb + kh * rp;
          for (int kw = 0; kw < d->k; kw++, wt += tapb) {
            const uint32_t* ep = ent + rq[kw];
            const uint32_t* ee = ent + rq[kw + 1];
            for (; ep < ee; ep++) {
              uint32_t e = *ep;
              int32_t v = e & 0xff;
              const int8_t* w = wt + (e >> 8);
              SA(0); SA(1); SA(2); SA(3); SA(4); SA(5); SA(6); SA(7);
              SA(8); SA(9); SA(10); SA(11); SA(12); SA(13); SA(14); SA(15);
            }
          }
        }
        SO(0); SO(1); SO(2); SO(3); SO(4); SO(5); SO(6); SO(7);
        SO(8); SO(9); SO(10); SO(11); SO(12); SO(13); SO(14); SO(15);
      }
    }
}
#endif

// ======================================================================== other operators
void maxpool3s2(const uint8_t* in, int h, int w, int c, uint8_t* out)
{
  int oh = h / 2, ow = w / 2, ipw = w + 2, opw = ow + 2, rs = ipw * c;
  for (int oy = 0; oy < oh; oy++)
    for (int ox = 0; ox < ow; ox++) {
      const uint8_t* r0 = in + ((2 * oy) * ipw + 2 * ox) * c;   // bordered coords, pad 1
      const uint8_t* r1 = r0 + rs;
      const uint8_t* r2 = r1 + rs;
      uint8_t* o = out + ((oy + 1) * opw + ox + 1) * c;
      for (int ch = 0; ch < c; ch++) {
        uint32_t m = r0[ch], v;
#define MX(p) v = (p); if (v > m) m = v
        MX(r0[c + ch]); MX(r0[2 * c + ch]);
        MX(r1[ch]); MX(r1[c + ch]); MX(r1[2 * c + ch]);
        MX(r2[ch]); MX(r2[c + ch]); MX(r2[2 * c + ch]);
#undef MX
        o[ch] = (uint8_t)m;
      }
    }
}

void add_relu(const add_desc_t* d, const int8_t* y, const void* res, uint8_t* out)
{
  int pw = d->w + 2, n = d->w * d->c;
#if !USE_XRQ
  int64_t rnd = (int64_t)1 << (d->k - 1);
#endif
  for (int row = 1; row <= d->h; row++) {
    int off = (row * pw + 1) * d->c;
    const int8_t* yp = y + off;
    uint8_t* op = out + off;
    if (d->res_signed) {
      const int8_t* rp = (const int8_t*)res + off;
      for (int i = 0; i < n; i++) {
        int64_t v = (int64_t)yp[i] * d->A + (int64_t)rp[i] * d->B;
#if USE_XRQ
        op[i] = (uint8_t)xrequant(v, d->rqw);
#else
        v = (v + rnd) >> d->k;
        if (v < 0) v = 0;
        if (v > 255) v = 255;
        if (v < d->T) v = 0;
        op[i] = (uint8_t)v;
#endif
      }
    } else {
      const uint8_t* rp = (const uint8_t*)res + off;
      for (int i = 0; i < n; i++) {
        int64_t v = (int64_t)yp[i] * d->A + (int64_t)rp[i] * d->B;
#if USE_XRQ
        op[i] = (uint8_t)xrequant(v, d->rqw);
#else
        v = (v + rnd) >> d->k;
        if (v < 0) v = 0;
        if (v > 255) v = 255;
        if (v < d->T) v = 0;
        op[i] = (uint8_t)v;
#endif
      }
    }
  }
}

void avgpool(const uint8_t* in, int h, int w, int c, uint8_t* out)
{
  int pw = w + 2;
  for (int ch = 0; ch < c; ch++) {
    int64_t s = 0;
    for (int y = 1; y <= h; y++)
      for (int x = 1; x <= w; x++) s += in[(y * pw + x) * c + ch];
#if USE_XRQ
    out[ch] = (uint8_t)xrequant(s, g_pool.rqw);
#else
    out[ch] = (uint8_t)rq_sw(s, g_pool.M, g_pool.sh, 0, 255, -1000);
#endif
  }
}

void fc(const fc_desc_t* d, const uint8_t* in, int32_t* out)
{
  const int32_t* bias = blob(d->bias);
#if CFG_NAIVE
  const int8_t* W = blob(d->w_naive);
  for (int o = 0; o < d->out; o++) {
    int64_t acc = bias[o];
    for (int i = 0; i < d->in; i++) acc += in[i] * W[o * d->in + i];
    out[o] = (int32_t)acc;
  }
#else
  const uint64_t* wp = blob(d->w_blk);
  const uint64_t* xv = (const uint64_t*)in;
  for (int o = 0; o < d->out; o += 4) {
    int64_t a0 = bias[o], a1 = bias[o + 1], a2 = bias[o + 2], a3 = bias[o + 3];
    for (int j = 0; j < d->in / 8; j++) {
#if USE_XQMAC
      uint64_t x = xv[j];
      a0 = xqmac8u(a0, x, wp[0]); a1 = xqmac8u(a1, x, wp[1]);
      a2 = xqmac8u(a2, x, wp[2]); a3 = xqmac8u(a3, x, wp[3]);
#else
      const uint8_t* xb = (const uint8_t*)(xv + j);
      const int8_t* wb = (const int8_t*)wp;
      _Pragma("GCC unroll 8") for (int b = 0; b < 8; b++) {
        int32_t v = xb[b];
        a0 += v * wb[b]; a1 += v * wb[8 + b]; a2 += v * wb[16 + b]; a3 += v * wb[24 + b];
      }
#endif
      wp += 4;
    }
    out[o] = (int32_t)a0; out[o + 1] = (int32_t)a1; out[o + 2] = (int32_t)a2; out[o + 3] = (int32_t)a3;
  }
#endif
}
