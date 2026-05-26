"""
deframer_profile.py — LC/FF predictor for deframer on iCE40 UP5K.

Loads measured synthesis results from profiles/sweep_deframer.csv.
Queries round up PacketLenElems to the next synthesized power of 2,
giving a conservative (never under) estimate.

Usage:
    from deframer_profile import predict, feasible
    lc, ff = predict(uw=1, pn=8, ple=600)
"""

import csv
import math
from pathlib import Path

LC_CAP = 5280

_data: dict[tuple, tuple] = {}   # (uw, pn, ple) -> (lc, ff)
_ple_vals: list[int] = []        # sorted synthesized PLE values (powers of 2)


def _load(path: Path = Path(__file__).parent / "profiles/sweep_deframer.csv") -> None:
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["UnpackedWidth"]), int(row["PackedNum"]), int(row["PacketLenElems"]))
            _data[key] = (int(row["LC"]), int(row["FF"]))
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
    Predict (LC, FF) for a deframer configuration.
    PacketLenElems is rounded up to the next synthesized power of 2.
    Raises AssertionError if uw * pn != 8.
    Raises ValueError if ple exceeds the sweep range or LC exceeds cap.
    """
    assert uw * pn == 8, f"UnpackedWidth={uw} * PackedNum={pn} must equal 8 (got {uw * pn})"
    ple_key = _ceil_ple(ple)
    key = (uw, pn, ple_key)
    if key not in _data:
        raise KeyError(f"No entry for (UW={uw}, PN={pn}, PLE={ple_key}) — run sweep_deframer.py first")
    lc, ff = _data[key]
    if lc > LC_CAP:
        raise ValueError(f"UW={uw} PN={pn} PLE={ple_key}: LC={lc} exceeds cap ({LC_CAP})")
    return lc, ff


def feasible(uw: int, pn: int, ple: int) -> bool:
    """Return True if the configuration fits within the LC cap."""
    try:
        predict(uw, pn, ple)
        return True
    except (AssertionError, KeyError, ValueError):
        return False


_load()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Query deframer LC/FF from sweep lookup")
    ap.add_argument("--uw",  type=int, required=True, help="UnpackedWidth")
    ap.add_argument("--pn",  type=int, required=True, help="PackedNum")
    ap.add_argument("--ple", type=int, required=True, help="PacketLenElems")
    args = ap.parse_args()

    try:
        lc, ff = predict(uw=args.uw, pn=args.pn, ple=args.ple)
        ple_key = _ceil_ple(args.ple)
        note = f" (rounded up from {args.ple})" if ple_key != args.ple else ""
        print(f"predict(uw={args.uw}, pn={args.pn}, ple={ple_key}{note})  =>  LC={lc}  FF={ff}")
    except (AssertionError, KeyError, ValueError) as e:
        print(f"Error: {e}")
