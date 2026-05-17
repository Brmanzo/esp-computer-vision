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
from model.protocol  import HEADER, TAIL, WAKEUP, IMG_BYTES, build_frame, parse_response

# ── Serial config ─────────────────────────────────────────────────────────────
BAUD = 115200  # 12 MHz / (prescale=13 * 8) ≈ 115385 baud (0.16% error)

# ── Globals shared between threads ───────────────────────────────────────────
hw_result: list = [None]   # hw_result[0] set by consumer thread

# ── Helpers ───────────────────────────────────────────────────────────────────

def producer(frame: bytes) -> None:
    ser.write(frame)


def consumer() -> None:
    """Read bytes from the FPGA and parse the frame."""
    # Read up to 4 bytes with a short inter-byte timeout so we don't block
    # for 30 s when the 0x99 wakeup was already flushed from the buffer.
    # The normal response is either [0x99, class, 0xA5, 0x5A] (4 bytes) or
    # [class, 0xA5, 0x5A] (3 bytes) when the wakeup was already consumed.
    raw = ser.read(3)           # minimum useful payload
    extra = ser.read(1)         # grab the 4th byte if it arrives within timeout
    raw = bytes(raw) + bytes(extra)

    print(f"  Raw bytes received : {' '.join(f'{x:02x}' for x in raw)}")

    try:
        hw_result[0] = parse_response(raw)
        print(f"  class_byte={hw_result[0]:#04x}  tail=a5 5a  ✓")
    except ValueError as e:
        print(f"  ✗ {e}")


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
frame = build_frame(pixels)
print(f"\nSending {len(frame)} bytes to {serial_port} @ {BAUD} baud...")

# 4. Open serial port, start producer and consumer threads
ser = serial.Serial(
    port          = serial_port,
    baudrate      = BAUD,
    parity        = serial.PARITY_NONE,
    stopbits      = serial.STOPBITS_ONE,
    bytesize      = serial.EIGHTBITS,
    timeout       = 5,     # 5 s for first 3 bytes (CNN pipeline + UART TX)
    inter_byte_timeout = 0.05,  # 50 ms gap ends the 4th-byte read immediately
    rtscts        = False, # FPGA rts_o=0 on reset → CTS=0 at host → deadlock if True
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
