#!/usr/bin/env python3
# Usage: python3 demo/python_demo.py <sample_idx> [ttyUSBx]
# Run from project root.

import os
import sys
import time
import serial
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.sample    import get_sample
from model.inference import get_inference, get_inference_from_pixels

# ── Protocol constants (must match deframer.sv / class_framer.sv) ────────────
HEADER    = bytes([0xA5, 0x5A])
TAIL      = bytes([0xA5, 0x5A])
WAKEUP    = 0x99
IMG_BYTES = (320 * 240) // 8   # 9600 — 8 binary pixels per byte, LSB-first

# ── Serial config ─────────────────────────────────────────────────────────────
BAUD = 115200  # 12 MHz / (prescale=13 * 8) ≈ 115385 baud (0.16% error)

# ── Globals shared between threads ───────────────────────────────────────────
hw_result: list = [None]   # hw_result[0] set by consumer thread

# ── Helpers ───────────────────────────────────────────────────────────────────

def pack_pixels(pixels: list) -> bytes:
    """Pack a flat list of binary {0,1} pixels into bytes, LSB-first."""
    buf = bytearray(IMG_BYTES)
    for i, px in enumerate(pixels):
        if px:
            buf[i // 8] |= 1 << (i % 8)
    return bytes(buf)


def producer(frame: bytes) -> None:
    ser.write(frame)


def consumer() -> None:
    """Read bytes from the FPGA and parse the frame."""
    # Worst-case frame: [0x99 wakeup] + [class_byte] + [0xA5] + [0x5A] = 4 bytes
    buf = ser.read(4)
    raw: list[int] = list(buf)

    print(f"  Raw bytes received : {' '.join(f'{x:02x}' for x in raw)}")

    # Scan for the response frame: skip leading 0x99 wakeup bytes, then
    # expect [class_byte, 0xA5, 0x5A]
    i = 0
    while i < len(raw) and raw[i] == WAKEUP:
        print(f"  (skipping wakeup 0x{WAKEUP:02x} at offset {i})")
        i += 1

    if i + 3 > len(raw):
        print(f"  Not enough bytes after wakeup skip (need 3, have {len(raw) - i}).")
        return

    class_byte = raw[i]
    tail_got   = bytes(raw[i+1:i+3])

    print(f"  class_byte={class_byte:#04x}  tail={tail_got.hex(' ')}", end="  ")

    if tail_got == TAIL:
        print("✓")
        hw_result[0] = class_byte
    else:
        print(f"✗ (expected {TAIL.hex(' ')})")


# ── Main ──────────────────────────────────────────────────────────────────────

if len(sys.argv) < 2:
    print("Usage: python3 demo/python_demo.py <sample_idx> [ttyUSBx]")
    sys.exit(1)

sample_idx  = int(sys.argv[1])
serial_port = "/dev/" + sys.argv[2] if len(sys.argv) > 2 else "/dev/ttyUSB2"

inject_env = os.environ.get("INJECT_PIXELS", "")
# 1. Load sample from dataset
if inject_env == "zeros":
    n_pixels = 320 * 240
    pixels = [0] * n_pixels
    label = None
    print(f"  Smoke test (all zeros) : {label}")
elif inject_env == "ones":
    n_pixels = 320 * 240
    pixels = [1] * n_pixels
    label = None
    print(f"  Smoke test (all ones) : {label}")
else:
    print(f"Loading sample {sample_idx} from dataset...")
    pixels, label = get_sample(sample_idx)
    if pixels is None:
        print("Failed to load sample.")
        sys.exit(1)
    print(f"  Ground truth label : {label}")

# 2. Hardware-accurate software inference
print("Running software inference...")
sw_pred = get_inference_from_pixels(pixels)
print(f"  SW prediction      : {sw_pred}  {'✓' if (label is not None and sw_pred == label) else ''}")

# 3. Build the UART frame: [0xA5, 0x5A] + 9600 image bytes
frame = HEADER + pack_pixels(pixels)
print(f"\nSending {len(frame)} bytes to {serial_port} @ {BAUD} baud...")

# 4. Open serial port, start producer and consumer threads
ser = serial.Serial(
    port     = serial_port,
    baudrate = BAUD,
    parity   = serial.PARITY_NONE,
    stopbits = serial.STOPBITS_ONE,
    bytesize = serial.EIGHTBITS,
    timeout  = 30,
    rtscts   = False,  # FPGA rts_o=0 on reset → CTS=0 at host → deadlock if True
)

producer_thread = threading.Thread(target=producer, args=(frame,), daemon=True)
consumer_thread = threading.Thread(target=consumer, daemon=True)

ser.reset_input_buffer()
ser.reset_output_buffer()
time.sleep(0.05)

t0 = time.monotonic()
producer_thread.start()
consumer_thread.start()

producer_thread.join()
consumer_thread.join()
ser.close()

tx_s = time.monotonic() - t0
print(f"  Done in {tx_s:.2f} s")

# 5. Report
if hw_result[0] is None:
    print("No valid response received from FPGA.")
    sys.exit(1)

hw_pred    = hw_result[0]
sw_match   = (hw_pred == sw_pred)

print()
if label is not None:
    hw_correct = hw_pred == label
    sw_correct = sw_pred == label
    print(f"  Ground truth  : {label}")
    print(f"  SW prediction : {sw_pred}  {'✓' if sw_correct else '✗'}")
    print(f"  HW prediction : {hw_pred}  {'✓' if hw_correct else '✗'}")
else:
    print(f"  Ground truth  : N/A (Smoke Test)")
    print(f"  SW prediction : {sw_pred}")
    print(f"  HW prediction : {hw_pred}")

print()
if sw_match:
    print("\033[92m[MATCH]    HW and SW agree\033[0m", end="  ")
else:
    print("\033[91m[MISMATCH] HW and SW disagree\033[0m", end="  ")

if label is not None:
    print("\033[92m[CORRECT]\033[0m" if hw_pred == label else "\033[91m[WRONG]\033[0m")
else:
    print()
