#!/usr/bin/env python3
"""
sweep_classifier_dsp_1_8.py — Characterise DSPCount LC savings for classifier_layer.

Loads a completed sweep_classifier.csv (DSPCount=0 runs), selects a few
representative (IC, CC) sample points per (TB, WB) corner at evenly-spaced
LC percentiles, then synthesises those points with every valid DSPCount
(1..min(ClassCount, 8)).

The DSP=0 baseline is re-used directly from the CSV — no re-synthesis needed.
Output CSV has the same columns as sweep_classifier.csv plus LC_dsp0, so the
two files can be concatenated for a unified corner analysis.

Usage (from repo root):
    python3 profiling/classifier_layer/sweep/sweep_classifier_dsp_1_8.py --base profiling/classifier_layer/profiles/sweep_classifier.csv
    python3 profiling/classifier_layer/sweep/sweep_classifier_dsp_1_8.py --base profiling/classifier_layer/profiles/sweep_classifier.csv --samples 5 --out classifier_dsp_chars.csv
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

LC_CAP  = 5280
DSP_CAP = 8
MODULE  = "classifier_layer"

FIXED = {
    "TermCount": 32,
    "BusBits":   8,
    "BiasBits":  8,
    "ShiftBits": 0,
}


def valid_dsp_counts(cc: int) -> list[int]:
    return [d for d in range(1, min(cc, DSP_CAP) + 1) if cc % d == 0]


def get_synth_sources(repo: str) -> list[str]:
    r = subprocess.run(
        ["python3", "sim/util/get_filelist.py", "rtl/top/top.json"],
        capture_output=True, text=True, cwd=repo,
    )
    return r.stdout.split()


def synthesize(repo, sources, tb, ic, cc, wb, dsp, yosys):
    params = {
        **FIXED,
        "DSPCount":   dsp,
        "TermBits":   tb,
        "InChannels": ic,
        "ClassCount": cc,
        "WeightBits": wb,
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
    return tot.get("LC", 0), tot.get("FF", 0)


def load_baseline(path: str) -> dict:
    """Load DSP=0 CSV. Returns {(tb, ic, cc, wb): (lc, ff)}."""
    baseline = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["DSPCount"]) != 0:
                continue
            key = (int(row["TermBits"]), int(row["InChannels"]),
                   int(row["ClassCount"]), int(row["WeightBits"]))
            baseline[key] = (int(row["LC"]), int(row["FF"]))
    return baseline


def select_samples(points: list[tuple], lc_vals: list[int], n: int) -> list[tuple]:
    """Pick n points at evenly-spaced LC percentiles."""
    if len(points) <= n:
        return points
    lc_arr = np.array(lc_vals, dtype=float)
    targets = np.linspace(lc_arr.min(), lc_arr.max(), n)
    chosen = set()
    for t in targets:
        idx = int(np.argmin(np.abs(lc_arr - t)))
        chosen.add(idx)
    return [points[i] for i in sorted(chosen)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base",    default="profiling/classifier_layer/profiles/sweep_classifier_dsp_0.csv")
    ap.add_argument("--out",     default="profiling/classifier_layer/profiles/classifier_dsp_chars.csv")
    ap.add_argument("--samples", default=6,      type=int,
                    help="(IC, CC) sample points per (TB, WB) corner (default 6)")
    ap.add_argument("--tb",      default=None,   type=int,
                    help="Only synthesize this TermBits value")
    ap.add_argument("--dsp",     default=None,   type=int,
                    help="Only synthesize this DSPCount value")
    ap.add_argument("--append",  action="store_true",
                    help="Append to existing output CSV instead of overwriting")
    ap.add_argument("--yosys",   default="yosys")
    args = ap.parse_args()

    repo     = os.getcwd()
    sources  = get_synth_sources(repo)
    baseline = load_baseline(args.base)

    # Group baseline points by (TB, WB) corner
    corners: dict[tuple, list] = defaultdict(list)
    for (tb, ic, cc, wb), (lc, ff) in baseline.items():
        if args.tb is not None and tb != args.tb:
            continue
        corners[(tb, wb)].append((ic, cc, lc, ff))

    runs = 0
    t0   = time.time()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    append = args.append and os.path.isfile(args.out)
    mode   = "a" if append else "w"
    with open(args.out, mode, newline="") as f:
        writer = csv.writer(f)
        if not append:
            writer.writerow(["TermBits", "InChannels", "ClassCount", "WeightBits",
                             "DSPCount", "LC", "FF", "LC_dsp0"])

        for (tb, wb), pts in sorted(corners.items()):
            if args.dsp is not None:
                pts = [p for p in pts if p[1] % args.dsp == 0]
            if not pts:
                print(f"  (no valid CC for DSP={args.dsp} in this corner, skipping)")
                continue
            lc_vals = [p[2] for p in pts]
            samples = select_samples(pts, lc_vals, args.samples)

            print(f"\n--- TB={tb} WB={wb}  "
                  f"({len(pts)} baseline pts, {len(samples)} sampled) ---")

            for ic, cc, lc_dsp0, ff_dsp0 in samples:
                if not append:
                    writer.writerow([tb, ic, cc, wb, 0, lc_dsp0, ff_dsp0, lc_dsp0])

                for dsp in valid_dsp_counts(cc):
                    if args.dsp is not None and dsp != args.dsp:
                        continue
                    lc, ff = synthesize(repo, sources, tb, ic, cc, wb, dsp, args.yosys)
                    runs += 1
                    elapsed = time.time() - t0
                    rate    = runs / elapsed if elapsed > 0 else 0
                    savings = (lc_dsp0 - lc) if lc is not None else None
                    print(f"  TB={tb} WB={wb} IC={ic:2d} CC={cc:2d} DSP={dsp}"
                          f"  LC={lc}  savings={savings}"
                          f"  [{runs} runs, {rate:.1f}/s]",
                          flush=True)

                    if lc is not None:
                        writer.writerow([tb, ic, cc, wb, dsp, lc, ff, lc_dsp0])
                f.flush()

    elapsed = time.time() - t0
    print(f"\nDone: {runs} synthesis runs in {elapsed/60:.1f} min")
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
