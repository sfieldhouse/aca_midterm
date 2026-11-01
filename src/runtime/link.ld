OUTPUT_ARCH("riscv")
ENTRY(_start)
SECTIONS
{
  . = 0x80000000;
  .text.init : { *(.text.init) }
  . = ALIGN(0x1000);
  .tohost : { *(.tohost) }
  . = ALIGN(0x1000);
  .text : { *(.text .text.*) }
  . = ALIGN(64);
  .rodata : { *(.rodata .rodata.* .srodata .srodata.*) }
  . = ALIGN(64);
  .data : { *(.data .data.* .sdata .sdata.*) }
  . = ALIGN(64);
  __bss_start = .;
  .bss : { *(.sbss .sbss.* .bss .bss.* COMMON) }
  . = ALIGN(64);
  __bss_end = .;
  . = . + 0x100000;
  __stack_top = .;
  _end = .;
}
