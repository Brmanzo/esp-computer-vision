#!/usr/bin/env python3
"""
sweep_pool.py — Sweep LC/FF usage for pool_layer on iCE40 UP5K.

Sweeps InBits (1-8), InChannels (1-32), and PoolMode (0=max, 1=avg).
InChannels == OutChannels and InBits == OutBits always.
KernelWidth is fixed at 2 (2x2 pooling), no DSPs.

Usage (from repo root):
    python3 profiling/pool_layer/sweep_pool.py
    python3 profiling/pool_layer/sweep_pool.py --ic_max 16 --ib_max 4
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
MODULE = "pool_layer"

FIXED = {
    "KernelWidth": 2,
    "LineWidthPx": 28,
    "LineCountPx": 28,
    "Unsigned":    0,
}


def get_synth_sources() -> list[str]:
    r = subprocess.run(
        ["python3", "sim/util/get_filelist.py", "rtl/top/top.json"],
        capture_output=True, text=True, cwd=os.getcwd(),
    )
    return r.stdout.split()


def synthesize(sources: list[str], ic: int, ib: int, mode: int, yosys: str) -> tuple:
    params = {
        **FIXED,
        "InChannels": ic,
        "InBits":     ib,
        "OutBits":    ib,
        "PoolMode":   mode,
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
    """Parse 'A:B' into range(A, B+1), or 'N' into range(N, N+1)."""
    if ":" in s:
        a, b = s.split(":")
        return range(int(a), int(b) + 1)
    n = int(s)
    return range(n, n + 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out",  default="profiling/pool_layer/profiles/sweep_pool.csv")
    ap.add_argument("--ib",   default="1:8",  metavar="A:B", help="InBits range (default 1:8)")
    ap.add_argument("--ic",   default="1:32", metavar="A:B", help="InChannels range (default 1:32)")
    ap.add_argument("--mode", default="0:1",  metavar="A:B", help="PoolMode range: 0=max 1=avg (default 0:1)")
    ap.add_argument("--yosys", default="yosys")
    args = ap.parse_args()

    sources = get_synth_sources()
    if not sources:
        print("ERROR: no source files found — run from repo root", file=sys.stderr)
        sys.exit(1)

    ib_range   = _range(args.ib)
    ic_range   = _range(args.ic)
    mode_range = _range(args.mode)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    runs = 0
    t0   = time.time()

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["InBits", "InChannels", "PoolMode", "LC", "FF"])

        for ib in ib_range:
            for mode in mode_range:
                mode_str = "max" if mode == 0 else "avg"
                for ic in ic_range:
                    lc, ff = synthesize(sources, ic, ib, mode, args.yosys)
                    runs  += 1
                    elapsed = time.time() - t0
                    rate    = runs / elapsed if elapsed > 0 else 0
                    print(f"  IB={ib} IC={ic:2d} mode={mode_str}"
                          f"  LC={lc}  FF={ff}"
                          f"  [{runs} runs, {rate:.1f}/s]",
                          flush=True)

                    if lc is None:
                        continue

                    writer.writerow([ib, ic, mode, lc, ff])
                    f.flush()

                    if lc > LC_CAP:
                        print(f"  (LC cap reached, stopping IB={ib} mode={mode_str})")
                        break

    elapsed = time.time() - t0
    print(f"\nDone: {runs} synthesis runs in {elapsed/60:.1f} min")
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
