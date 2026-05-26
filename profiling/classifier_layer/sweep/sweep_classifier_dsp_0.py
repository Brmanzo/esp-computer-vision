#!/usr/bin/env python3
"""
sweep_classifier.py — Sweep LC/FF usage for classifier_layer on iCE40 UP5K.

Sweeps TermBits, InChannels, ClassCount, WeightBits with DSPCount=0.
Fixed: TermCount=32, BusBits=8, BiasBits=8, ShiftBits=0.

Usage (from repo root):
    python3 profiling/classifier_layer/sweep/sweep_classifier.py
    python3 profiling/classifier_layer/sweep/sweep_classifier.py --tb 4:4 --cc 1:8
"""

import argparse
import csv
import os
import subprocess
import sys
import time

sys.path.insert(0, os.getcwd())
from nn.util import parse, total_cells, _fold, _base  # noqa: E402

LC_CAP = 5280
MODULE = "classifier_layer"

FIXED = {
    "TermCount":  32,
    "BusBits":    8,
    "BiasBits":   8,
    "ShiftBits":  0,
}


def get_synth_sources() -> list[str]:
    r = subprocess.run(
        ["python3", "sim/util/get_filelist.py", "rtl/top/top.json"],
        capture_output=True, text=True, cwd=os.getcwd(),
    )
    return r.stdout.split()


def synthesize(sources: list[str], tb: int, ic: int, cc: int, wb: int, dsp: int, yosys: str) -> tuple:
    params = {
        **FIXED,
        "TermBits":   tb,
        "InChannels": ic,
        "ClassCount": cc,
        "WeightBits": wb,
        "DSPCount":   dsp,
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
    return tot.get("LC", 0), tot.get("FF", 0)


def _range(s: str) -> range:
    if ":" in s:
        a, b = s.split(":")
        return range(int(a), int(b) + 1)
    n = int(s)
    return range(n, n + 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out",   default="profiling/classifier_layer/profiles/sweep_classifier.csv")
    ap.add_argument("--tb",    default="1:8",  metavar="A:B", help="TermBits range (default 1:8)")
    ap.add_argument("--ic",    default="1:16", metavar="A:B", help="InChannels range (default 1:16)")
    ap.add_argument("--cc",    default="1:16", metavar="A:B", help="ClassCount range (default 1:16)")
    ap.add_argument("--wb",    default="2:8",  metavar="A:B", help="WeightBits range (default 2:8)")
    ap.add_argument("--dsp",   default="0",    metavar="A:B", help="DSPCount range (default 0)")
    ap.add_argument("--yosys", default="yosys")
    args = ap.parse_args()

    sources = get_synth_sources()
    if not sources:
        print("ERROR: no source files found — run from repo root", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    runs = 0
    t0   = time.time()

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["TermBits", "InChannels", "ClassCount", "WeightBits", "DSPCount", "LC", "FF"])

        for tb in _range(args.tb):
            for wb in _range(args.wb):
                for dsp in _range(args.dsp):
                    for ic in _range(args.ic):
                        for cc in _range(args.cc):
                            lc, ff = synthesize(sources, tb, ic, cc, wb, dsp, args.yosys)
                            runs  += 1
                            elapsed = time.time() - t0
                            rate    = runs / elapsed if elapsed > 0 else 0
                            print(f"  TB={tb} WB={wb} DSP={dsp} IC={ic:2d} CC={cc:2d}"
                                  f"  LC={lc}  FF={ff}"
                                  f"  [{runs} runs, {rate:.1f}/s]",
                                  flush=True)

                            if lc is None:
                                continue

                            writer.writerow([tb, ic, cc, wb, dsp, lc, ff])
                            f.flush()

                            if lc > LC_CAP:
                                print(f"  (LC cap at CC={cc}, stopping TB={tb} WB={wb} DSP={dsp} IC={ic})")
                                break

    elapsed = time.time() - t0
    print(f"\nDone: {runs} synthesis runs in {elapsed/60:.1f} min")
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
