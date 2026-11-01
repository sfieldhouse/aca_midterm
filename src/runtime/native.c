// Host (x86) shim: lets the bare-metal sources run natively for fast checks.
#include <stdint.h>
uint64_t rdinstret(void) { return 0; }
