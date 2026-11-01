// Bare-metal ResNet-18 benchmark: runs the embedded images, checks the logits
// against the Python integer reference (golden), reports retired instructions.
//
// Output lines (parsed by scripts/run_benchmarks.py):
//   IMG <i> instret <n> pred <c> label <l> golden <OK|MISMATCH>
//   OP <name> <total instret over all images>
//   NNZ <conv index> <nonzeros> <elements>
//   RESULT <PASS|FAIL> images <n> total_instret <n>
#include "nn.h"
#include "runtime.h"
#include "xqnn.h"

#ifndef NIMG
#define NIMG 1
#endif

extern const int g_nimg;
extern const int g_labels[];
static int32_t logits[NCLASS];

#ifdef NATIVE
#include <stdio.h>
#include <stdlib.h>
static uint8_t* load(const char* dir, const char* f, long* sz)
{
  char path[1024];
  snprintf(path, sizeof path, "%s/%s", dir, f);
  FILE* fp = fopen(path, "rb");
  if (!fp) { perror(path); exit(2); }
  fseek(fp, 0, SEEK_END);
  *sz = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  uint8_t* p = aligned_alloc(64, (*sz + 63) / 64 * 64);
  if (fread(p, 1, *sz, fp) != (size_t)*sz) exit(3);
  fclose(fp);
  return p;
}
const uint8_t* g_blob;
int32_t xq_acc_emu[32 * 8];
int main(int argc, char** argv)
{
  if (argc < 2) { fprintf(stderr, "usage: %s <export dir> [nimg]\n", argv[0]); return 2; }
  long sz, isz, gsz;
  g_blob = load(argv[1], "model.bin", &sz);
  const uint8_t* images = load(argv[1], "images.bin", &isz);
  const int32_t* golden = (const int32_t*)load(argv[1], "golden.bin", &gsz);
  int n = isz / IMG_BYTES;
  if (argc > 2 && atoi(argv[2]) < n) n = atoi(argv[2]);
#else
extern const uint8_t model_bin[], images_bin[], golden_bin[];
const uint8_t* g_blob = model_bin;
int main(void)
{
  const uint8_t* images = images_bin;
  const int32_t* golden = (const int32_t*)golden_bin;
  int n = NIMG < g_nimg ? NIMG : g_nimg;
#endif
  int ok_all = 1;
  uint64_t total = 0;
  for (int i = 0; i < n; i++) {
    uint64_t t0 = rdinstret();
    model_run((const int8_t*)(images + (uint64_t)i * IMG_BYTES), logits);
    uint64_t t = rdinstret() - t0;
    total += t;
    int ok = 1, pred = 0;
    for (int c = 0; c < NCLASS; c++) {
      if (logits[c] != golden[(uint64_t)i * NCLASS + c]) ok = 0;
      if (logits[c] > logits[pred]) pred = c;
    }
    ok_all &= ok;
#ifdef NATIVE
    int label = -1;
#else
    int label = g_labels[i];
#endif
    printf("IMG %d instret %lu pred %d label %d golden %s\n", i, t, pred, label, ok ? "OK" : "MISMATCH");
  }
  for (int k = 0; k < g_nops; k++) printf("OP %s %lu\n", g_op_name[k], g_op_instret[k]);
  for (int k = 0; k < 20; k++)
    if (g_nin[k]) printf("NNZ %d %lu %lu\n", k, g_nnz[k], g_nin[k]);
  printf("RESULT %s images %d total_instret %lu\n", ok_all ? "PASS" : "FAIL", n, total);
  return ok_all ? 0 : 1;
}
