// HTIF (tohost/fromhost) based console + exit, and a small printf.
#include "runtime.h"
#include <stdarg.h>

volatile uint64_t tohost __attribute__((section(".tohost"), aligned(64)));
volatile uint64_t fromhost __attribute__((section(".tohost"), aligned(64)));

static void htif_send(uint64_t cmd)
{
  while (tohost)
    fromhost = 0;
  tohost = cmd;
}

void putch(int c)
{
  // device 1 (console), command 1 (putchar)
  htif_send((1ull << 56) | (1ull << 48) | (uint8_t)c);
}

void exit(int code)
{
  htif_send(((uint64_t)code << 1) | 1);
  for (;;) {}
}

void* memset(void* d, int c, size_t n)
{
  uint8_t* p = d;
  while (n--) *p++ = (uint8_t)c;
  return d;
}

void* memcpy(void* d, const void* s, size_t n)
{
  uint8_t* p = d;
  const uint8_t* q = s;
  while (n--) *p++ = *q++;
  return d;
}

static void print_u(uint64_t v, int base, int width, char pad)
{
  char buf[24];
  int i = 0;
  do {
    int d = v % base;
    buf[i++] = d < 10 ? '0' + d : 'a' + d - 10;
    v /= base;
  } while (v);
  while (i < width) buf[i++] = pad;
  while (i) putch(buf[--i]);
}

int printf(const char* fmt, ...)
{
  va_list ap;
  va_start(ap, fmt);
  for (; *fmt; fmt++) {
    if (*fmt != '%') { putch(*fmt); continue; }
    fmt++;
    char pad = ' ';
    int width = 0, lng = 0;
    if (*fmt == '0') { pad = '0'; fmt++; }
    while (*fmt >= '0' && *fmt <= '9') width = width * 10 + (*fmt++ - '0');
    while (*fmt == 'l') { lng++; fmt++; }
    switch (*fmt) {
      case 'd': {
        int64_t v = lng ? va_arg(ap, int64_t) : va_arg(ap, int);
        if (v < 0) { putch('-'); v = -v; if (width) width--; }
        print_u((uint64_t)v, 10, width, pad);
        break;
      }
      case 'u': print_u(lng ? va_arg(ap, uint64_t) : va_arg(ap, unsigned), 10, width, pad); break;
      case 'x': print_u(lng ? va_arg(ap, uint64_t) : va_arg(ap, unsigned), 16, width, pad); break;
      case 's': { const char* s = va_arg(ap, const char*); while (*s) putch(*s++); break; }
      case 'c': putch(va_arg(ap, int)); break;
      case '%': putch('%'); break;
      default: putch('?');
    }
  }
  va_end(ap);
  return 0;
}
