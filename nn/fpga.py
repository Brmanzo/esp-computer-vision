#!/usr/bin/env python3
# nn.fpga.py
# Usage: cnn.py fpga <sample_idx> [ttyUSBx] [--trials N]

import os
import sys
import time
import random
import serial
import threading
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nn.inference  import get_inference_from_pixels
from nn.uart   import build_frame, parse_response
from nn.globals    import NN_CFG, BAUD, DATAPATH, prepare_data

# ── Serial config ─────────────────────────────────────────────────────────────

def _find_icebreaker_port() -> str | None:
    """Return the ttyUSBN name for the iCEBreaker UART via /dev/serial/by-id symlinks."""
    by_id = Path("/dev/serial/by-id")
    if not by_id.exists():
        return None
    candidates = [l for l in sorted(by_id.iterdir()) if "ice" in l.name.lower()]
    # FT2232H: if01 = UART, if00 = JTAG — prefer UART
    for suffix in ("if01", "if00"):
        for link in candidates:
            if suffix in link.name:
                return link.resolve().name
    return candidates[0].resolve().name if candidates else None

# ── Per-trial send/receive ─────────────────────────────────────────────────────

def run_trial(ser: serial.Serial, pixels: list) -> int | None:
    """Send one frame over UART and return the decoded HW class byte, or None."""
    hw_result: list[int | None] = [None]

    def producer():
        ser.write(build_frame(pixels))

    def consumer():
        from nn.uart import TAIL
        raw = bytearray()
        while True:
            b = ser.read(1)
            if not b:
                break
            raw.extend(b)
            if len(raw) >= 2 and raw[-2:] == TAIL:
                break
                
        print(f"  Raw bytes: {' '.join(f'{x:02x}' for x in raw)}")
        try:
            hw_result[0] = parse_response(raw)
            print(f"  class_byte={hw_result[0]:#04x}  tail=a5 5a  ✓")
        except ValueError as e:
            print(f"  ✗ {e}")

    ser.reset_input_buffer()
    ser.reset_output_buffer()
    time.sleep(0.05)

    t0 = time.monotonic()
    pt = threading.Thread(target=producer, daemon=True)
    ct = threading.Thread(target=consumer, daemon=True)
    pt.start(); ct.start()
    pt.join(); ct.join()
    print(f"  Done in {time.monotonic() - t0:.2f} s")
    return hw_result[0]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        usage="cnn.py fpga <sample_idx> [ttyUSBx] [--trials N]"
    )
    parser.add_argument("sample_idx", type=int, nargs="?", default=0,
                        help="Dataset index for single-shot or seed trial")
    parser.add_argument("port",       nargs="?", default=None,
                        help="Serial device name (e.g. ttyUSB2); auto-detected if omitted")
    parser.add_argument("--trials",   type=int,  default=1,
                        help="Run N trials over random indices; break on SW/HW mismatch")
    args = parser.parse_args()

    port = args.port or _find_icebreaker_port() or "ttyUSB2"
    if args.port is None:
        print(f"Auto-detected port: {port}")
    serial_port = "/dev/" + port
    ser = serial.Serial(
        port               = serial_port,
        baudrate           = BAUD,
        parity             = serial.PARITY_NONE,
        stopbits           = serial.STOPBITS_ONE,
        bytesize           = serial.EIGHTBITS,
        timeout            = 5,
        inter_byte_timeout = 0.05,
        rtscts             = False,
    )

    inject_env = os.environ.get("INJECT_PIXELS", "")

    cfg = NN_CFG
    assert cfg.in_dims.height is not None and cfg.in_dims.width is not None
    n_pixels = cfg.in_dims.width * cfg.in_dims.height

    # Load dataset once upfront (skipped for smoke-test modes).
    samples: list[tuple[list, int]] = []
    if not inject_env:
        print("Loading dataset...")
        _, test_loader, _ = prepare_data(DATAPATH, cfg.in_dims.height, cfg.in_dims.width, 1)
        for batch_img, batch_label in test_loader:
            pixels = (batch_img[0].flatten() > 0.5).int().tolist()
            samples.append((pixels, int(batch_label[0])))
        print(f"  {len(samples)} test samples loaded.\n")

    # For multi-trial mode pick random indices; single-shot uses sample_idx directly.
    if args.trials == 1:
        indices = [args.sample_idx]
    else:
        indices = random.sample(range(len(samples)), min(args.trials, len(samples)))

    matches = mismatches = 0

    for trial_num, idx in enumerate(indices, 1):
        if args.trials > 1:
            print(f"\n{'='*60}")
            print(f"Trial {trial_num}/{args.trials}  sample_idx={idx}")
            print(f"{'='*60}")

        # 1. Load sample
        if inject_env == "zeros":
            pixels, label = [0] * n_pixels, None
            print("  Smoke test (all zeros)")
        elif inject_env == "ones":
            pixels, label = [1] * n_pixels, None
            print("  Smoke test (all ones)")
        else:
            pixels, label = samples[idx]
            print(f"  Ground truth : {label}")

        # 2. Software inference
        sw_pred = get_inference_from_pixels(pixels)
        print(f"  SW prediction: {sw_pred}  {'✓' if label is not None and sw_pred == label else ''}")

        # 3. Hardware inference
        print(f"Sending frame to {serial_port} @ {BAUD} baud...")
        hw_pred = run_trial(ser, pixels)

        if hw_pred is None:
            print("  No valid response from FPGA. Stopping.\n  Did you make bitstream ESP=0 ?")
            break

        # 4. Report
        sw_match = hw_pred == sw_pred
        print()
        if label is not None:
            print(f"  Ground truth : {label}")
            print(f"  SW           : {sw_pred}  {'✓' if sw_pred == label else '✗'}")
            print(f"  HW           : {hw_pred}  {'✓' if hw_pred == label else '✗'}")

        if sw_match:
            matches += 1
            print("\033[92m[MATCH]    HW and SW agree\033[0m")
        else:
            mismatches += 1
            print("\033[91m[MISMATCH] HW and SW disagree\033[0m")
            break

    ser.close()

    if args.trials > 1:
        total = matches + mismatches
        print(f"\n{'='*60}")
        print(f"Trials: {total}  Match: {matches}  Mismatch: {mismatches}")


if __name__ == "__main__":
    main()
