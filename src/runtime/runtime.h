// Bare-metal runtime for Spike: HTIF console/exit, printf, instret counter.
#ifndef RUNTIME_H
#define RUNTIME_H
#include <stdint.h>
#include <stddef.h>

#ifdef __riscv
static inline uint64_t rdinstret(void)
{
  uint64_t v;
  __asm__ volatile("csrr %0, minstret" : "=r"(v));
  return v;
}
#else
uint64_t rdinstret(void);
#endif

void putch(int c);
int printf(const char* fmt, ...);
void exit(int code) __attribute__((noreturn));
void* memset(void* d, int c, size_t n);
void* memcpy(void* d, const void* s, size_t n);
#endif
