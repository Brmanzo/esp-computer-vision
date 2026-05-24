"""
conv_profile.py — LC and FF cost predictor for conv_layer on iCE40 UP5K.

Loads regression coefficients from profile_coeffs.csv (exported from MATLAB).
Models per (InBits, WeightBits, DSPCount) corner:
    LC = A*OC*IC + B*OC + C*IC + D
    FF = A*OC*log2(IC) + B*OC + C*IC + D

Usage:
    from profile import predict, feasible, frontier
    lc, ff = predict(ic=4, oc=8, ib=4, wb=4, dsp=0)
    configs = frontier(ib=4, wb=4, dsp=0)
"""

import csv
import math
import os
from pathlib import Path

LC_CAP = 5280
FF_CAP = 5280   # iCE40 UP5K: one FF per LC, same pool
DSP_CAP = 8

_LC: dict[tuple, tuple] = {}   # (ib, wb, dsp) -> (A, B, C, D, R2)
_FF: dict[tuple, tuple] = {}


def _load(path: Path=(Path(__file__).parent / "../profiles/profile_coeffs.csv")) -> None:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "profile_coeffs.csv")
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["InBits"]), int(row["WeightBits"]), int(row["DSPCount"]))
            coeffs = (float(row["A"]), float(row["B"]),
                      float(row["C"]), float(row["D"]), float(row["R2"]))
            if row["Model"] == "LC":
                _LC[key] = coeffs
            else:
                _FF[key] = coeffs


def _lc(ic: int, oc: int, ib: int, wb: int, dsp: int) -> float:
    # DSP>0 corners are keyed by (ib, -1, dsp) — WB doesn't affect LC in DSP mode
    key = (ib, -1, dsp) if dsp > 0 else (ib, wb, dsp)
    if key not in _LC:
        raise KeyError(f"No LC model for InBits={ib} WeightBits={wb} DSPCount={dsp}")
    A, B, C, D, _ = _LC[key]
    return A*oc*ic + B*oc + C*ic + D


def _ff(ic: int, oc: int, ib: int, wb: int) -> float:
    key = (ib, wb, 0)
    if key not in _FF:
        raise KeyError(f"No FF model for InBits={ib} WeightBits={wb}")
    A, B, C, D, _ = _FF[key]
    return A*oc*math.log2(max(ic, 1)) + B*oc + C*ic + D


def predict(ic: int, oc: int, ib: int, wb: int,
            dsp: int = 0) -> tuple[float, float]:
    """
    Predict (LC, FF) for a conv_layer configuration.
    Raises ValueError if either exceeds the iCE40 UP5K cap.
    """
    if dsp > 0 and oc % dsp != 0:
        raise ValueError(f"DSPCount={dsp} does not divide OutChannels={oc}")

    lc = _lc(ic, oc, ib, wb, dsp)
    ff = _ff(ic, oc, ib, wb)

    errors = []
    if lc > LC_CAP:
        errors.append(f"LC={lc:.0f} exceeds cap ({LC_CAP})")
    if ff > FF_CAP:
        errors.append(f"FF={ff:.0f} exceeds cap ({FF_CAP})")
    if errors:
        raise ValueError(
            f"IC={ic} OC={oc} IB={ib} WB={wb} DSP={dsp}: " + ", ".join(errors)
        )
    return lc, ff


def feasible(ic: int, oc: int, ib: int, wb: int, dsp: int = 0) -> bool:
    """Return True if the configuration fits within both caps."""
    try:
        predict(ic, oc, ib, wb, dsp)
        return True
    except (ValueError, KeyError):
        return False


def valid_dsp_counts(oc: int) -> list[int]:
    """Divisors of oc that are <= DSP_CAP."""
    return [d for d in range(1, min(oc, DSP_CAP) + 1) if oc % d == 0]


def frontier(ib: int, wb: int, dsp: int = 0,
             ic_max: int = 32, oc_max: int = 32) -> list[tuple[int, int, float, float]]:
    """
    All feasible (IC, OC) pairs for a corner, returned as
    [(ic, oc, predicted_lc, predicted_ff), ...] sorted by lc ascending.
    """
    results = []
    for ic in range(1, ic_max + 1):
        for oc in range(1, oc_max + 1):
            try:
                lc, ff = predict(ic, oc, ib, wb, dsp)
                results.append((ic, oc, lc, ff))
            except (ValueError, KeyError):
                break   # LC monotone in OC for DSP=0; safe to break
    return sorted(results, key=lambda x: x[2])


# Load on import
_load()


if __name__ == "__main__":
    print(f"Corners loaded — LC: {len(_LC)}  FF: {len(_FF)}")
    for ib, wb, dsp in sorted(_LC):
        print(f"  LC corner IB={ib} WB={wb} DSP={dsp}")

    print()
    test_cases = [
        dict(ic=4, oc=8,  ib=4, wb=4, dsp=0), # exceeds cap
        dict(ic=4, oc=8,  ib=4, wb=4, dsp=2), # fits
        dict(ic=4, oc=8,  ib=4, wb=4, dsp=3), # 8 % 3 != 0 — expect error
        dict(ic=4, oc=16, ib=4, wb=4, dsp=8), # Missing corner
        dict(ic=4, oc=9,  ib=2, wb=8, dsp=3),
    ]
    for kw in test_cases:
        try:
            lc, ff = predict(**kw)
            print(f"  predict({kw})  =>  LC={lc:.0f}  FF={ff:.0f}")
        except (ValueError, KeyError) as e:
            print(f"  predict({kw})  =>  {e}")
