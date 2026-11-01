// ResNet-18 graph driver: static buffers, per-operator instret accounting.
#include "nn.h"
#include "runtime.h"
#include "xqnn.h"

#define BUF_BYTES (114 * 114 * 64)        // largest bordered activation (stem output)
#define MAX_NNZ (56 * 56 * 64)            // largest compressible conv input
#define MAX_ROWS (58 * 58 + 1)

static uint8_t bufs[5][BUF_BYTES] __attribute__((aligned(64)));
#if CFG_SPARSE
static uint32_t csr_ent[MAX_NNZ] __attribute__((aligned(64)));
static uint32_t csr_row[MAX_ROWS] __attribute__((aligned(64)));
#endif
static uint8_t pooled[512] __attribute__((aligned(64)));

const char* g_op_name[N_OPS];
uint64_t g_op_instret[N_OPS];
int g_nops;
uint64_t g_nnz[20], g_nin[20];

static int op_idx;
static uint64_t op_t0;
static inline void op_begin(const char* name)
{
  g_op_name[op_idx] = name;
  op_t0 = rdinstret();
}
static inline void op_end(void)
{
  g_op_instret[op_idx++] += rdinstret() - op_t0;
}

#if CFG_SPARSE
static const char* cname[20] = {
  "csr:stem", "csr:layer1.0.conv1", "csr:layer1.0.conv2", "csr:layer1.1.conv1", "csr:layer1.1.conv2",
  "csr:layer2.0.conv1", "csr:layer2.0.downsample.0", "csr:layer2.0.conv2", "csr:layer2.1.conv1",
  "csr:layer2.1.conv2", "csr:layer3.0.conv1", "csr:layer3.0.downsample.0", "csr:layer3.0.conv2",
  "csr:layer3.1.conv1", "csr:layer3.1.conv2", "csr:layer4.0.conv1", "csr:layer4.0.downsample.0",
  "csr:layer4.0.conv2", "csr:layer4.1.conv1", "csr:layer4.1.conv2"};
static int csr_valid_for = -1;   // conv index whose input/OC the current CSR encodes
#endif

// Run conv `ci` reading `in` (bordered), writing bordered `out`.
static void run_conv(int ci, const uint8_t* in, uint8_t* out)
{
  const conv_desc_t* d = &g_convs[ci];
  zero_border(out, d->oh, d->ow, d->oc);
#if CFG_NAIVE
  op_begin(d->name); conv_naive(d, in, out); op_end();
#elif !CFG_SPARSE
  op_begin(d->name); conv_dense(d, in, out); op_end();
#else
  if (ci == 0) { op_begin(d->name); conv_dense(d, in, out); op_end(); return; }
  uint32_t nnz;
  // downsample conv shares the block input and OC with conv1 -> reuse its CSR
  int reuse = csr_valid_for >= 0 && g_convs[csr_valid_for].oc == d->oc &&
              d->k == 1 && csr_valid_for == ci - 1;
  if (!reuse) {
    op_begin(cname[ci]);
    nnz = csr_compress(in, d->ih, d->iw, d->ic, d->oc, csr_ent, csr_row);
    op_end();
  } else {
    op_begin(cname[ci]); op_end();
    nnz = csr_row[(d->ih + 2) * (d->iw + 2)];
  }
  csr_valid_for = ci;
  g_nnz[ci] += nnz;
  g_nin[ci] += (uint64_t)d->ih * d->iw * d->ic;
  op_begin(d->name);
  if ((uint64_t)nnz * 1000 > (uint64_t)SPARSE_DISPATCH_PERMILLE * d->ih * d->iw * d->ic)
    conv_dense(d, in, out);
  else
    conv_sparse(d, csr_ent, csr_row, out);
  op_end();
#endif
}

void model_run(const int8_t* img, int32_t* logits)
{
  static const char* add_name[8] = {"add:layer1.0", "add:layer1.1", "add:layer2.0", "add:layer2.1",
                                    "add:layer3.0", "add:layer3.1", "add:layer4.0", "add:layer4.1"};
  op_idx = 0;
#if CFG_SPARSE
  csr_valid_for = -1;
#endif
  uint8_t *x = bufs[0], *t1 = bufs[1], *t2 = bufs[2], *rs = bufs[3], *y = bufs[4];
  run_conv(0, (const uint8_t*)img, t1);
  zero_border(x, 56, 56, 64);
  op_begin("maxpool"); maxpool3s2(t1, 112, 112, 64, x); op_end();
  int ci = 1;
  for (int b = 0; b < 8; b++) {
    int c1 = ci++;
    int ds = (b == 2 || b == 4 || b == 6) ? ci++ : -1;
    int c2 = ci++;
    run_conv(c1, x, t1);
    if (ds >= 0) run_conv(ds, x, rs);
    run_conv(c2, t1, t2);
    const add_desc_t* a = &g_adds[b];
    zero_border(y, a->h, a->w, a->c);
    op_begin(add_name[b]); add_relu(a, (const int8_t*)t2, ds >= 0 ? rs : x, y); op_end();
    uint8_t* t = x; x = y; y = t;
  }
  op_begin("avgpool"); avgpool(x, 7, 7, 512, pooled); op_end();
  op_begin("fc"); fc(&g_fc, pooled, logits); op_end();
  g_nops = op_idx;
}
