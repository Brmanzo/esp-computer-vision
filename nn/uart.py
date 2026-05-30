#!/usr/bin/env python3
# nn.uart.py
# Bradley Manzo 2026

# Shared UART framing constants and helpers used by python_demo.py and
# the cnn_uart cocotb integration test. Both must stay in sync with
# deframer.sv and class_framer.sv.

from nn.globals import NN_CFG

HEADER    = bytes([0xA5, 0x5A])
TAIL      = bytes([0xA5, 0x5A])
WAKEUP    = 0x99

def build_frame(pixels: list) -> bytes:
    """Pack a flat list of binary {0,1} pixels into the wire frame.

    Returns HEADER + 9600 image bytes (LSB-first packed).
    """
    buf = bytearray(NN_CFG.in_dims.term_count // 8)
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
    if len(raw) < 3:
        raise ValueError(f"Response too short (need at least 3 bytes): {list(raw)}")
    
    tail_got = bytes(raw[-2:])
    if tail_got != TAIL:
        raise ValueError(f"Tail mismatch: expected {list(TAIL)}, got {list(tail_got)}")
        
    class_id = raw[-3]
    return class_id
