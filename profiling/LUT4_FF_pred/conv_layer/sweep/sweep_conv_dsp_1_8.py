#!/usr/bin/env python3
"""
sweep_dsp.py — Characterise DSPCount LUT4 savings using existing DSP=0 data.

Loads a completed sweep_conv.csv state exploration (DSPCount=0 runs),
selects a few representative (IC, OC) sample points per (IB, WB) corner at 
evenly-spaced LUT4 percentiles, then synthesises those points with every 
valid DSPCount (divisors of OC <= 8).

The DSP=0 baseline is re-used directly from the CSV — no re-synthesis needed.
Output CSV has the same columns as sweep5d.csv plus DSPCount, so the two files
can be concatenated in MATLAB for a unified corner analysis.

Usage (from repo root):
    python3 sweep_dsp.py --base sweep5d.csv
    python3 sweep_dsp.py --base sweep5d.csv --samples 5 --ob 4 --out dsp_chars.csv
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nn.util import parse, total_cells, _fold, _base  # noqa: E402

from nn.globals import DSP_CAP    # iCE40 caps
MODULE  = "conv_layer"
KERNEL  = 3

FIXED = {
    "Stride":      1,
    "Padding":     1,
    "ShiftBits":   0,
    "KernelWidth": KERNEL,
    "BiasBits":    8,
    "LineWidthPx": 28,
    "LineCountPx": 28,
}


def valid_dsp_counts(oc: int) -> list[int]:
    return [d for d in range(1, min(oc, DSP_CAP) + 1) if oc % d == 0]


def get_synth_sources(repo: str) -> list[str]:
    r = subprocess.run(
        ["python3", "sim/util/get_filelist.py", "rtl/top/top.json"],
        capture_output=True, text=True, cwd=repo,
    )
    return r.stdout.split()


def synthesize(repo, sources, ic, oc, ib, wb, ob, dsp, yosys):
    params = {
        **FIXED,
        "DSPCount":    dsp,
        "InChannels":  ic,
        "OutChannels": oc,
        "InBits":      ib,
        "OutBits":     ob,
        "WeightBits":  wb,
    }
    param_cmds = "".join(f"chparam -set {k} {v} {MODULE}; " for k, v in params.items())
    script = (
        f"read_verilog -sv -DSYNTHESIS {' '.join(sources)}; "
        f"{param_cmds}"
        f"synth_ice40 -dsp -noflatten -top {MODULE}; stat"
    )
    r = subprocess.run([yosys, "-p", script], capture_output=True, text=True, cwd=repo)
    mods = parse((r.stdout + r.stderr).splitlines())
    top_key = next((k for k in mods if _base(k) == MODULE), None)
    if top_key is None:
        return None, None
    tot = _fold(dict(total_cells(top_key, mods, {})))
    return tot.get("LUT4", 0), tot.get("FF", 0)


def load_baseline(path: str) -> dict:
    """Load DSP=0 CSV. Returns {(ic,oc,ib,wb,ob): (lut4,ff)}."""
    baseline = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["InCh"]), int(row["OutCh"]),
                   int(row["InBits"]), int(row["WeightBits"]), int(row["OutBits"]))
            baseline[key] = (int(row["LUT4"]), int(row["FF"]))
    return baseline


def select_samples(points: list[tuple], lut4_vals: list[int], n: int) -> list[tuple]:
    """Pick n points at evenly-spaced LUT4 percentiles."""
    if len(points) <= n:
        return points
    lut4_arr = np.array(lut4_vals, dtype=float)
    targets = np.linspace(lut4_arr.min(), lut4_arr.max(), n)
    chosen = set()
    for t in targets:
        idx = int(np.argmin(np.abs(lut4_arr - t)))
        chosen.add(idx)
    return [points[i] for i in sorted(chosen)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base",    required=True,  help="Existing DSP=0 CSV (sweep5d.csv)")
    ap.add_argument("--out",     default="dsp_chars.csv")
    ap.add_argument("--ob",      default=None,   type=int,
                    help="Fix OutBits to one value (recommended; omit to use all in base)")
    ap.add_argument("--samples", default=6,      type=int,
                    help="(IC,OC) sample points per (IB,WB) corner (default 6)")
    ap.add_argument("--ib",      default=None,   type=int,
                    help="Only synthesize this InBits value")
    ap.add_argument("--dsp",     default=None,   type=int,
                    help="Only synthesize this DSPCount value")
    ap.add_argument("--append",  action="store_true",
                    help="Append to existing output CSV instead of overwriting")
    ap.add_argument("--yosys",   default="yosys")
    args = ap.parse_args()

    repo     = os.getcwd()
    sources  = get_synth_sources(repo)
    baseline = load_baseline(args.base)

    # Group baseline points by (IB, WB, OB) corner
    corners: dict[tuple, list] = defaultdict(list)
    for (ic, oc, ib, wb, ob), (lut4, ff) in baseline.items():
        if args.ob  is not None and ob  != args.ob:
            continue
        if args.ib  is not None and ib  != args.ib:
            continue
        corners[(ib, wb, ob)].append((ic, oc, lut4, ff))

    runs = 0
    t0   = time.time()

    append  = args.append and os.path.isfile(args.out)
    mode    = "a" if append else "w"
    with open(args.out, mode, newline="") as f:
        writer = csv.writer(f)
        if not append:
            writer.writerow(["InCh", "OutCh", "InBits", "WeightBits", "OutBits",
                             "DSPCount", "LUT4", "FF", "LUT4_dsp0"])

        for (ib, wb, ob), pts in sorted(corners.items()):
            # When targeting a specific DSP, restrict to OC values it can divide
            if args.dsp is not None:
                pts = [p for p in pts if p[1] % args.dsp == 0]
            if not pts:
                print(f"  (no OC values divisible by DSP={args.dsp} in this corner, skipping)")
                continue
            lut4_vals = [p[2] for p in pts]
            # HARDCODED MISSING POINTS LOGIC
            # Now targeting InCh=1 and InCh=4 to fill out their missing curves
            samples = [p for p in pts if p[1] == 24 and p[0] in [1, 4]]

            print(f"\n--- IB={ib} WB={wb} OB={ob}  "
                  f"({len(pts)} baseline pts, {len(samples)} missing targets) ---")

            missing_dsps = {
                1: [1, 2, 3, 4, 6],
                4: [1, 2, 3, 4, 6]
            }

            for ic, oc, lut4_dsp0, ff_dsp0 in samples:
                if not append:
                    writer.writerow([ic, oc, ib, wb, ob, 0, lut4_dsp0, ff_dsp0, lut4_dsp0])

                for dsp in valid_dsp_counts(oc):
                    if dsp == 0:
                        continue
                    if args.dsp is not None and dsp != args.dsp:
                        continue
                    # Only synthesize the exact missing DSP counts
                    if dsp not in missing_dsps.get(ic, []):
                        continue
                    lut4, ff = synthesize(repo, sources, ic, oc, ib, wb, ob, dsp, args.yosys)
                    runs += 1
                    elapsed = time.time() - t0
                    rate    = runs / elapsed if elapsed > 0 else 0
                    savings = (lut4_dsp0 - lut4) if lut4 is not None else None
                    print(f"  IC={ic:2d} OC={oc:2d} DSP={dsp}"
                          f"  LUT4={lut4}  savings={savings}"
                          f"  [{runs} runs, {rate:.1f}/s]",
                          flush=True)

                    if lut4 is not None:
                        writer.writerow([ic, oc, ib, wb, ob, dsp, lut4, ff, lut4_dsp0])
                f.flush()

    elapsed = time.time() - t0
    print(f"\nDone: {runs} synthesis runs in {elapsed/60:.1f} min")
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
