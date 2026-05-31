"""
deframer_profile.py — LUT4/FF predictor for deframer on iCE40 Icebreaker V1.1a.

Loads measured synthesis results from profiles/sweep_deframer.csv.
Queries round up PacketLenElems to the next synthesized power of 2,
giving a conservative (never under) estimate.

Usage:
    from deframer_profile import predict, feasible
    lut4, ff = predict(uw=1, pn=8, ple=600)
"""

import csv
import math
from pathlib import Path

from nn.globals import LC_CAP, BUS_WIDTH

_data: dict[tuple, tuple] = {}   # (uw, pn, ple) -> (lut4, ff)
_ple_vals: list[int] = []        # sorted synthesized PLE values (powers of 2)


def _load(path: Path = Path(__file__).parent / "profiles/sweep_deframer.csv") -> None:
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["UnpackedWidth"]), int(row["PackedNum"]), int(row["PacketLenElems"]))
            _data[key] = (int(row["LUT4"]), int(row["FF"]))
    global _ple_vals
    _ple_vals = sorted({ple for _, _, ple in _data})


def _ceil_ple(ple: int) -> int:
    """Return the smallest synthesized PLE value >= ple."""
    for p in _ple_vals:
        if p >= ple:
            return p
    raise ValueError(f"PacketLenElems={ple} exceeds max synthesized value ({_ple_vals[-1]})")


def predict(uw: int, pn: int, ple: int) -> tuple[int, int]:
    """
    Predict (LUT4, FF) for a deframer configuration.
    PacketLenElems is rounded up to the next synthesized power of 2.
    Raises AssertionError if uw * pn != BUS_WIDTH.
    Raises ValueError if ple exceeds the sweep range or LUT4 exceeds cap.
    """
    assert uw * pn == BUS_WIDTH, f"UnpackedWidth={uw} * PackedNum={pn} must equal 8 (got {uw * pn})"
    ple_key = _ceil_ple(ple)
    key = (uw, pn, ple_key)
    if key not in _data:
        raise KeyError(f"No entry for (UW={uw}, PN={pn}, PLE={ple_key}) — run sweep_deframer.py first")
    lut4, ff = _data[key]
    if lut4 > LC_CAP:
        raise ValueError(f"UW={uw} PN={pn} PLE={ple_key}: LUT4={lut4} exceeds cap ({LC_CAP})")
    return lut4, ff


def feasible(uw: int, pn: int, ple: int) -> bool:
    """Return True if the configuration fits within the LUT4 cap."""
    try:
        predict(uw, pn, ple)
        return True
    except (AssertionError, KeyError, ValueError):
        return False


_load()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Query deframer LUT4/FF from sweep lookup")
    ap.add_argument("--uw",  type=int, required=True, help="UnpackedWidth")
    ap.add_argument("--pn",  type=int, required=True, help="PackedNum")
    ap.add_argument("--ple", type=int, required=True, help="PacketLenElems")
    args = ap.parse_args()

    try:
        lut4, ff = predict(uw=args.uw, pn=args.pn, ple=args.ple)
        ple_key = _ceil_ple(args.ple)
        note = f" (rounded up from {args.ple})" if ple_key != args.ple else ""
        print(f"predict(uw={args.uw}, pn={args.pn}, ple={ple_key}{note})  =>  LUT4={lut4}  FF={ff}")
    except (AssertionError, KeyError, ValueError) as e:
        print(f"Error: {e}")
