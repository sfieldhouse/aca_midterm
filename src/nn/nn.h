// Quantized ResNet-18 bare-metal inference: data structures and kernel API.
#ifndef NN_H
#define NN_H
#include <stdint.h>

// ---------------------------------------------------------------- configuration
// CFG selects the kernel set (see README / DESIGN.md):
//   0 B0  naive dense, scalar MAC, SW requant
//   1 B1  optimized dense (2px x 4oc blocking), scalar MAC, SW requant
//   2 X1  B1 + xqmac8/xqmac8u
//   3 X2  X1 + xrequant
//   4 S0  sparse-input conv (runtime CSR), scalar MAC, SW requant
//   5 S1  S0 with xsqmac8 accumulator file (sparse custom instruction)
//   6 S2  S1 + xqmac8 (dense layers) + xrequant  ("all")
#ifndef CFG
#define CFG 6
#endif
#define CFG_NAIVE (CFG == 0)
#define CFG_SPARSE (CFG >= 4)
#define USE_XQMAC (CFG == 2 || CFG == 3 || CFG == 6)
#define USE_XRQ (CFG == 3 || CFG == 6)
#define USE_XSMAC (CFG == 5 || CFG == 6)
// S2: layers whose input density (permille) exceeds this use the dense xqmac8 kernel
#ifndef SPARSE_DISPATCH_PERMILLE
#define SPARSE_DISPATCH_PERMILLE 1001
#endif

// ---------------------------------------------------------------- descriptors
// All activation tensors are NHWC with a 1-pixel zero border ("bordered").
// The stem input is a pre-padded image [230][232][4] int8 (pad 3, 4th channel 0).
typedef struct {
  const char* name;
  int ic, oc;          // ic = padded channel count of the input buffer
  int ih, iw, oh, ow;  // logical (unpadded) sizes
  int k, stride, pad;
  int kwp;             // kernel width as stored (stem: 8, else k)
  int L;               // bytes per kernel row = kwp * ic
  int in_rowpix;       // pixels per row in the input buffer
  int in_off;          // input-buffer origin offset (pixels) = border - pad
  int mode, T;         // requant: mode 1 = uint8 ReLU/FATReLU(T), 0 = int8
  int in_signed;       // stem input is int8; other inputs uint8
  uint32_t w_naive;    // int8 [oc][k][L]
  uint32_t w_blk;      // int8 [oc/4][k][L/8][4][8]
  uint32_t w_sp;       // int8 [k][k][ic][oc]  (0 for stem)
  uint32_t bias;       // int32 [oc]
  uint32_t mult;       // int32 [oc]
  uint32_t shift;      // int32 [oc]
  uint32_t rqw;        // uint64 [oc]  xrequant parameter words
} conv_desc_t;

typedef struct {
  int h, w, c;
  int A, B, k, T;
  int res_signed;      // residual operand is int8 (downsample output) instead of uint8
  uint64_t rqw;        // xrequant word (M = 1, sh = k, T, mode 1)
} add_desc_t;

typedef struct {
  int M, sh;
  uint64_t rqw;
} pool_desc_t;

typedef struct {
  int in, out;
  uint32_t w_naive;    // int8 [out][in]
  uint32_t w_blk;      // int8 [out/4][in/8][4][8]
  uint32_t bias;       // int32 [out]
} fc_desc_t;

extern const conv_desc_t g_convs[20];
extern const add_desc_t g_adds[8];
extern const pool_desc_t g_pool;
extern const fc_desc_t g_fc;
extern const uint8_t* g_blob;   // weights/params blob

#define IMG_BYTES (230 * 232 * 4)
#define NCLASS 1000

// ---------------------------------------------------------------- kernels
void conv_naive(const conv_desc_t* d, const void* in, void* out);
void conv_dense(const conv_desc_t* d, const void* in, void* out);
void conv_sparse(const conv_desc_t* d, const uint32_t* ent, const uint32_t* rowptr, void* out);
uint32_t csr_compress(const uint8_t* in, int h, int w, int c, int oc, uint32_t* ent, uint32_t* rowptr);
void maxpool3s2(const uint8_t* in, int h, int w, int c, uint8_t* out);
void add_relu(const add_desc_t* d, const int8_t* y, const void* r, uint8_t* out);
void avgpool(const uint8_t* in, int h, int w, int c, uint8_t* out);
void fc(const fc_desc_t* d, const uint8_t* in, int32_t* out);
void zero_border(void* buf, int h, int w, int c);

// ---------------------------------------------------------------- model
#define N_OPS 64
extern const char* g_op_name[N_OPS];
extern uint64_t g_op_instret[N_OPS];
extern int g_nops;
extern uint64_t g_nnz[20], g_nin[20];   // conv-input nonzeros / elements (sparse configs)
void model_run(const int8_t* img, int32_t* logits);

static inline const void* blob(uint32_t off) { return g_blob + off; }
#endif
