#!/usr/bin/env python3
"""
sweep_deframer.py — Sweep LUT4/FF usage for deframer on iCE40 UP5K.

Sweeps PacketLenElems over powers of 2 for every valid (UnpackedWidth, PackedNum) pair.
Fixed constraint: UnpackedWidth * PackedNum == 8 (bus width).

Usage (from repo root):
    python3 profiling/overhead/deframer/sweep/sweep_deframer.py
    python3 profiling/overhead/deframer/sweep/sweep_deframer.py --ple 1:256
"""

import argparse
import csv
import math
import os
import subprocess
import sys
import time

sys.path.insert(0, os.getcwd())
from nn.util import parse, total_cells, _fold, _base  # noqa: E402

LC_CAP = 5280
MODULE = "deframer"

def get_synth_sources() -> list[str]:
    r = subprocess.run(
        ["python3", "sim/util/get_filelist.py", "rtl/top/top.json"],
        capture_output=True, text=True, cwd=os.getcwd(),
    )
    return r.stdout.split()


def synthesize(sources: list[str], uw: int, pn: int, ple: int, yosys: str) -> tuple:
    params = {
        "UnpackedWidth":  uw,
        "PackedNum":      pn,
        "PacketLenElems": ple,
    }
    param_cmds = "".join(f"chparam -set {k} {v} {MODULE}; " for k, v in params.items())
    script = (
        f"read_verilog -sv -DSYNTHESIS {' '.join(sources)}; "
        f"{param_cmds}"
        f"synth_ice40 -noflatten -top {MODULE}; stat"
    )
    r = subprocess.run([yosys, "-p", script], capture_output=True, text=True)
    mods    = parse((r.stdout + r.stderr).splitlines())
    top_key = next((k for k in mods if _base(k) == MODULE), None)
    if top_key is None:
        return None, None
    tot = _fold(dict(total_cells(top_key, mods, {})))
    return tot.get("LUT4", 0), tot.get("FF", 0)


def _range(s: str) -> range:
    """Parse 'A:B' into range(A, B+1), or 'N' into range(N, N+1)."""
    if ":" in s:
        a, b = s.split(":")
        return range(int(a), int(b) + 1)
    n = int(s)
    return range(n, n + 1)

valid_uw_pn = [(1, 8), (2, 4), (4, 2), (8, 1)]

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out",  default="profiling/overhead/deframer/profiles/sweep_deframer.csv")
    ap.add_argument("--ple", default="0:1024",  metavar="A:B", help="PacketLenElems range: (default 0:1024)")
    ap.add_argument("--yosys", default="yosys")
    args = ap.parse_args()

    sources = get_synth_sources()
    if not sources:
        print("ERROR: no source files found — run from repo root", file=sys.stderr)
        sys.exit(1)

    ple_range      = _range(args.ple)
    ple_max        = max(ple_range) if ple_range else 1
    ple_full_range = [2**b for b in range(int(math.log2(ple_max)) + 1)]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    runs = 0
    t0   = time.time()

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["UnpackedWidth", "PackedNum", "PacketLenElems", "LUT4", "FF"])

        for valid_uw, valid_pn in valid_uw_pn:
            for valid_ple in ple_full_range:
                lut4, ff = synthesize(sources, valid_uw, valid_pn, valid_ple, args.yosys)
                runs  += 1
                elapsed = time.time() - t0
                rate    = runs / elapsed if elapsed > 0 else 0
                print(f"  UnpackedWidth={valid_uw} PackedNum={valid_pn} PacketLenElems={valid_ple}"
                        f"  LUT4={lut4}  FF={ff}"
                        f"  [{runs} runs, {rate:.1f}/s]",
                        flush=True)

                if lut4 is None:
                    continue

                writer.writerow([valid_uw, valid_pn, valid_ple, lut4, ff])
                f.flush()

                if lut4 > LC_CAP:
                    print(f"  (LUT4 cap reached, stopping UnpackedWidth={valid_uw} PackedNum={valid_pn} PacketLenElems={valid_ple})")
                    break

    elapsed = time.time() - t0
    print(f"\nDone: {runs} synthesis runs in {elapsed/60:.1f} min")
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
