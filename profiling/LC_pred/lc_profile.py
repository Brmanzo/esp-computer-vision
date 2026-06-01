"""
interconnect_profile.py — Predict actual ICESTORM_LC from profile_model pred_ff.

Regression over bitstream sweep results shows that full-design LC cost is
driven almost entirely by the predicted FF count (R² ≈ 0.97):

    act_lc = A · pred_ff + B

The pred_lut4 coefficient is ~0 because DSPs absorb MAC arithmetic, leaving
routing/glue LUT4s that scale with pipeline register count rather than
arithmetic complexity.

Coefficients are loaded from profiles/interconnect_coeffs.csv, produced by
profiling/LC_pred/regression/lc_regression.m.

Usage:
    from profiling.LC_pred.lc_profile import predict_lc, FF_CAP
    estimated_lc = predict_lc(pred_ff)
    feasible     = pred_ff <= FF_CAP
"""

import csv
from pathlib import Path

from nn.globals import LC_CAP

_A: float = 0.0
_B: float = 0.0
_R2: float = 0.0


def _load(path: Path = Path(__file__).parent / "profiles/interconnect_coeffs.csv") -> None:
    global _A, _B, _R2
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            term = row["Term"]
            val  = float(row["Value"])
            if term == "A":
                _A = val
            elif term == "B":
                _B = val
            elif term == "R2":
                _R2 = val


def predict_lc(pred_ff: float) -> float:
    """Return estimated ICESTORM_LC given a predicted FF count."""
    return _A * pred_ff + _B


# Pre-computed FF threshold: pred_ff <= FF_CAP implies predict_lc(pred_ff) <= LC_CAP.
# Use this as the DFS feasibility check instead of pred_lut4 <= LC_CAP - LC_HEADROOM.
FF_CAP: float = 0.0


def _init() -> None:
    global FF_CAP
    _load()
    if _A == 0.0:
        raise RuntimeError(
            "interconnect_coeffs.csv loaded but coefficient A is zero — "
            "regenerate it by running:\n"
            "  matlab -batch \"cd('profiling/LC_pred/regression'); lc_regression\""
        )
    FF_CAP = (LC_CAP - _B) / _A


_init()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Predict ICESTORM_LC from pred_ff")
    ap.add_argument("--pred-ff", type=float, required=True,
                    help="Predicted FF count from profile_model")
    args = ap.parse_args()

    lc = predict_lc(args.pred_ff)
    feasible = args.pred_ff <= FF_CAP
    print(f"pred_ff={args.pred_ff:.0f}  =>  predicted LC={lc:.0f}  "
          f"({'OK' if feasible else 'EXCEEDS CAP'}  FF_CAP={FF_CAP:.0f})")
    print(f"Model: act_lc = {_A:.4f}·pred_ff + {_B:.2f}   R²={_R2:.4f}")
