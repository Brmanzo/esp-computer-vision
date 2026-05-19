#!/usr/bin/env python3
# nn.protocol.py
# Bradley Manzo 2026

# Shared UART framing constants and helpers used by python_demo.py and
# the cnn_uart cocotb integration test. Both must stay in sync with
# deframer.sv and class_framer.sv.

HEADER    = bytes([0xA5, 0x5A])
TAIL      = bytes([0xA5, 0x5A])
WAKEUP    = 0x99
IMG_BYTES = (320 * 240) // 8   # 9600 — 8 binary pixels per byte, LSB-first


def build_frame(pixels: list) -> bytes:
    """Pack a flat list of binary {0,1} pixels into the wire frame.

    Returns HEADER + 9600 image bytes (LSB-first packed).
    """
    buf = bytearray(IMG_BYTES)
    for i, px in enumerate(pixels):
        if px:
            buf[i // 8] |= 1 << (i % 8)
    return HEADER + bytes(buf)


def parse_response(raw: bytes) -> int:
    """Parse the 4-byte FPGA response and return the class ID.

    Skips any leading WAKEUP (0x99) bytes, then expects:
      [class_byte, 0xA5, 0x5A]

    Raises ValueError on malformed or truncated input.
    """
    i = 0
    while i < len(raw) and raw[i] == WAKEUP:
        i += 1
    if i + 3 > len(raw):
        raise ValueError(
            f"Not enough bytes after wakeup skip (need 3, have {len(raw) - i}): {list(raw)}"
        )
    class_id = raw[i]
    tail_got = bytes(raw[i + 1 : i + 3])
    if tail_got != TAIL:
        raise ValueError(
            f"Tail mismatch: expected {list(TAIL)}, got {list(tail_got)}"
        )
    return class_id
