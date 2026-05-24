#!/usr/bin/env python3
"""
sweep_conv.py — Synthesize conv_layer across the parameter space and stop each
OutCh sweep as soon as the actual LC count exceeds the iCE40 UP5K cap.

DSPCount=0 mode (default): sweeps InCh, OutCh, InBits, OutBits, WeightBits.
DSP mode (--dsp):          sweeps the same space but for each OutCh inserts
                           synthesis runs for every valid DSPCount, i.e. the
                           divisors of OutCh that are <= DSP_CAP (8).
                           Early-break on OutCh uses the max valid DSPCount
                           (minimum possible LC) as the probe.

Fixed params: Stride=1, Padding=1, ShiftBits=0, KernelWidth=3,
              BiasBits=8, LineWidthPx=28, LineCountPx=28

Usage (from repo root):
    python3 sweep5d.py                          # DSPCount=0 only
    python3 sweep5d.py --dsp                    # all valid DSP counts per OC
    python3 sweep5d.py --ib 4:4 --wb 4:4 --dsp
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nn.util import parse, total_cells, _fold, _base  # noqa: E402

LC_CAP  = 5280
DSP_CAP = 8         # iCE40 UP5K SB_MAC16 blocks
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
    """Divisors of oc that are <= DSP_CAP, in ascending order."""
    return [d for d in range(1, min(oc, DSP_CAP) + 1) if oc % d == 0]


def get_synth_sources(repo: str) -> list[str]:
    r = subprocess.run(
        ["python3", "sim/util/get_filelist.py", "rtl/top/top.json"],
        capture_output=True, text=True, cwd=repo,
    )
    return r.stdout.split()


def synthesize(repo, sources, ic, oc, ib, wb, ob, dsp, yosys) -> tuple[int | None, int | None]:
    """Run one yosys synthesis and return (LC, FF). Returns (None, None) on error."""
    # Weights and Biases are intentionally omitted: chparam on a vector whose
    # width depends on other parameters causes a yosys segfault.  The default
    # Weights='0 does not collapse the multiplier tree because the weights_i
    # port is connected structurally through filter.sv and is not fully
    # constant-propagated by the optimizer.
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
    return tot.get("LC", 0), tot.get("FF", 0)


def _range(s: str) -> range:
    a, b = s.split(":")
    return range(int(a), int(b) + 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out",   default="sweep5d.csv")
    ap.add_argument("--ib",    default="2:8",  metavar="A:B", help="InBits range")
    ap.add_argument("--wb",    default="2:8",  metavar="A:B", help="WeightBits range")
    ap.add_argument("--ob",    default="2:8",  metavar="A:B", help="OutBits range")
    ap.add_argument("--ic",    default="1:32", metavar="A:B", help="InChannels range")
    ap.add_argument("--oc",    default="1:32", metavar="A:B", help="OutChannels range")
    ap.add_argument("--dsp",   action="store_true",
                               help="Sweep valid DSPCounts per OutCh instead of fixing DSPCount=0")
    ap.add_argument("--yosys", default="yosys")
    args = ap.parse_args()

    repo       = os.path.dirname(os.path.abspath(__file__))
    sources    = get_synth_sources(repo)
    ib_range   = _range(args.ib)
    wb_range   = _range(args.wb)
    ob_range   = _range(args.ob)
    ic_range   = _range(args.ic)
    oc_range   = _range(args.oc)

    naive_total = (len(ib_range) * len(wb_range) * len(ob_range)
                   * len(ic_range) * len(oc_range))

    runs = 0
    t0   = time.time()

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["InCh", "OutCh", "InBits", "WeightBits", "OutBits", "DSPCount", "LC", "FF"])

        for ib in ib_range:
          for wb in wb_range:
            for ob in ob_range:
              for ic in ic_range:

                # Probe OC=1, DSP=1 (only valid count for OC=1).
                # LC is monotone in InCh so break the IC loop if this fails.
                lc, ff = synthesize(repo, sources, ic, 1, ib, wb, ob, 1 if args.dsp else 0, args.yosys)
                runs += 1
                elapsed = time.time() - t0
                rate    = runs / elapsed if elapsed > 0 else 0
                print(f"  IC={ic:2d} OC= 1 DSP={1 if args.dsp else 0}"
                      f"  IB={ib} WB={wb} OB={ob}"
                      f"  LC={lc}  FF={ff}"
                      f"  [{runs} runs, {rate:.1f}/s]",
                      flush=True)

                if lc is None or lc > LC_CAP:
                    break  # InCh break: all larger InCh also infeasible

                writer.writerow([ic, 1, ib, wb, ob, 1 if args.dsp else 0, lc, ff])
                f.flush()

                for oc in oc_range:
                    if oc == 1:
                        continue

                    dsps = valid_dsp_counts(oc) if args.dsp else [0]

                    if not args.dsp:
                        # DSPCount=0: LC is monotone in OC, safe to break early.
                        lc, ff = synthesize(repo, sources, ic, oc, ib, wb, ob, 0, args.yosys)
                        runs += 1
                        elapsed = time.time() - t0
                        rate    = runs / elapsed if elapsed > 0 else 0
                        print(f"  IC={ic:2d} OC={oc:2d} DSP=0"
                              f"  IB={ib} WB={wb} OB={ob}"
                              f"  LC={lc}  FF={ff}"
                              f"  [{runs} runs, {rate:.1f}/s]",
                              flush=True)

                        if lc is None or lc > LC_CAP:
                            break  # monotone: no larger OC can fit

                        writer.writerow([ic, oc, ib, wb, ob, 0, lc, ff])
                        f.flush()
                    else:
                        # DSP mode: max valid DSPCount varies with OC (e.g. OC=13
                        # only gets DSP=1 but OC=16 gets DSP=8), so LC is NOT
                        # monotone in OC.  No early break — sweep all OC values.
                        for dsp in dsps:
                            lc, ff = synthesize(repo, sources, ic, oc, ib, wb, ob, dsp, args.yosys)
                            runs += 1
                            elapsed = time.time() - t0
                            rate    = runs / elapsed if elapsed > 0 else 0
                            print(f"  IC={ic:2d} OC={oc:2d} DSP={dsp}"
                                  f"  IB={ib} WB={wb} OB={ob}"
                                  f"  LC={lc}  FF={ff}"
                                  f"  [{runs} runs, {rate:.1f}/s]",
                                  flush=True)

                            if lc is not None and lc <= LC_CAP:
                                writer.writerow([ic, oc, ib, wb, ob, dsp, lc, ff])
                        f.flush()

    elapsed = time.time() - t0
    print(f"\nDone: {runs} synthesis runs in {elapsed/60:.1f} min"
          f"  (naive grid: {naive_total},"
          f" pruned {100*(1 - runs/naive_total):.1f}%)")
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
