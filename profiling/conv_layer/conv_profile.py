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

from nn.globals import LC_CAP, DSP_CAP   # iCE40 caps

_LC: dict[tuple, tuple] = {}   # (ib, wb, dsp) -> (A, B, C, D, R2)
_FF: dict[tuple, tuple] = {}


def _load(path: Path=(Path(__file__).parent / "profiles/profile_coeffs.csv")) -> None:
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
        if dsp > 0:
            raise KeyError(
                f"No LC model for InBits={ib} DSPCount={dsp} — "
                f"run fill_corner(ib={ib}, dsp={dsp}) to synthesize it"
            )
        raise KeyError(f"No LC model for InBits={ib} WeightBits={wb} DSPCount={dsp}")
    A, B, C, D, _ = _LC[key]
    return A*oc*ic + B*oc + C*ic + D


def _ff(ic: int, oc: int, ib: int, wb: int) -> float:
    key = (ib, wb, 0)
    if key not in _FF:
        raise KeyError(f"No FF model for InBits={ib} WeightBits={wb}")
    A, B, C, D, _ = _FF[key]
    return A*oc*math.log2(max(ic, 1)) + B*oc + C*ic + D


def predict_conv_layer(ic: int, oc: int, ib: int, wb: int,
            dsp: int = 0) -> tuple[float, float]:
    """
    Predict (LC, FF) for a conv_layer configuration.
    If the (ib, dsp) corner is missing, synthesizes it automatically then retries.
    Raises ValueError if either resource exceeds the iCE40 UP5K cap.
    """
    if dsp > 0 and oc % dsp != 0:
        raise ValueError(f"DSPCount={dsp} does not divide OutChannels={oc}")

    try:
        lc = _lc(ic, oc, ib, wb, dsp)
    except KeyError:
        if dsp == 0:
            raise
        fill_corner(ib=ib, dsp=dsp)
        lc = _lc(ic, oc, ib, wb, dsp)

    ff = _ff(ic, oc, ib, wb)

    errors = []
    if lc > LC_CAP:
        errors.append(f"LC={lc:.0f} exceeds cap ({LC_CAP})")
    if ff > LC_CAP:
        errors.append(f"FF={ff:.0f} exceeds cap ({LC_CAP})")
    if errors:
        raise ValueError(
            f"IC={ic} OC={oc} IB={ib} WB={wb} DSP={dsp}: " + ", ".join(errors)
        )
    return math.ceil(lc), math.ceil(ff)


def feasible(ic: int, oc: int, ib: int, wb: int, dsp: int = 0) -> bool:
    """Return True if the configuration fits within both caps."""
    try:
        predict_conv_layer(ic, oc, ib, wb, dsp)
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
                lc, ff = predict_conv_layer(ic, oc, ib, wb, dsp)
                results.append((ic, oc, lc, ff))
            except (ValueError, KeyError):
                break   # LC monotone in OC for DSP=0; safe to break
    return sorted(results, key=lambda x: x[2])


def _count_csv_rows(path: Path, ib: int, dsp: int) -> int:
    if not path.exists():
        return 0
    with open(path, newline="") as f:
        return sum(1 for row in csv.DictReader(f)
                   if int(row["InBits"]) == ib and int(row["DSPCount"]) == dsp)


def fill_corner(ib: int, dsp: int,
                base_csv: str | None = None,
                samples: int = 8,
                yosys: str = "yosys") -> None:
    """
    Synthesize a missing (ib, dsp) LC corner, refit the regression, and reload.

    Appends new synthesis rows to profiles/sweep_conv_dsp_1-8.csv, reruns the
    MATLAB regression to overwrite profiles/profile_coeffs.csv, then reloads
    coefficients into this process.

    base_csv: path to DSP=0 sweep CSV used for sampling (IC, OC) points.
              Defaults to profiles/sweep_conv_dsp_0.csv.
    """
    import subprocess

    here    = Path(__file__).parent
    repo    = here.parent.parent          # esp-computer-vision/
    sweep   = here / "sweep/sweep_conv_dsp_1_8.py"
    regr    = here / "regression/conv_layer_dsp_1_8.m"
    dsp_csv = here / "profiles/sweep_conv_dsp_1-8.csv"

    if base_csv is None:
        base_csv = str(here / "profiles/sweep_conv_dsp_0.csv")

    def _run(label: str, cmd: list[str], cwd: Path = repo) -> str:
        print(f"[fill_corner] {label} ...", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
        if r.returncode != 0:
            print(r.stdout[-3000:])
            print(r.stderr[-3000:])
            raise RuntimeError(f"fill_corner: {label} failed (exit {r.returncode})")
        return r.stdout

    rows_before = _count_csv_rows(dsp_csv, ib=ib, dsp=dsp)
    _run(f"Synthesizing IB={ib} DSP={dsp} ({samples} points)",
         ["python3", str(sweep),
          "--base",    base_csv,
          "--ib",      str(ib),
          "--dsp",     str(dsp),
          "--samples", str(samples),
          "--out",     str(dsp_csv),
          "--append",
          "--yosys",   yosys])
    rows_after = _count_csv_rows(dsp_csv, ib=ib, dsp=dsp)
    print(f"[fill_corner] Synthesized {rows_after - rows_before} new rows "
          f"(total IB={ib} DSP={dsp} rows: {rows_after})", flush=True)
    if rows_after < 4:
        print(f"[fill_corner] WARNING: only {rows_after} rows — need ≥4 for regression fit. "
              f"Try --samples with a larger value or check baseline has OC divisible by {dsp}.")

    # MATLAB relative paths are resolved from cwd, not the script location,
    # so run from the regression/ directory where ../profiles/ resolves correctly.
    out = _run("Refitting regression",
               ["matlab", "-batch", f"run('{regr.resolve()}')"],
               cwd=regr.parent)
    if "Exported" in out or "Written" in out:
        for line in out.splitlines():
            if "corner" in line.lower() or "exported" in line.lower() or "written" in line.lower():
                print(f"[fill_corner] MATLAB: {line.strip()}")

    _LC.clear()
    _FF.clear()
    _load()
    key = (ib, -1, dsp)
    if key in _LC:
        print(f"[fill_corner] Done — corner {key} loaded successfully")
    else:
        print(f"[fill_corner] Corner {key} still missing after refit.")
        print(f"[fill_corner] Available LC corners: {sorted(_LC)}")


# Load on import
_load()


def _report() -> None:
    print(f"Corners loaded — LC: {len(_LC)}  FF: {len(_FF)}")
    dsp0 = [(ib, wb, dsp) for ib, wb, dsp in sorted(_LC) if dsp == 0]
    dspN = [(ib, wb, dsp) for ib, wb, dsp in sorted(_LC) if dsp  > 0]
    for ib, wb, dsp in dsp0:
        print(f"  LC  IB={ib} WB={wb}   DSP=0")
    for ib, wb, dsp in dspN:
        print(f"  LC  IB={ib} WB=any  DSP={dsp}")
    for ib, wb, _ in sorted(_FF):
        print(f"  FF  IB={ib} WB={wb}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Query or fill conv_layer LC/FF models")
    ap.add_argument("--fill", nargs=2, type=int, metavar=("IB", "DSP"),
                    help="Synthesize and fit a missing DSP corner, then report")
    ap.add_argument("--samples", type=int, default=8,
                    help="Sample points per corner when filling (default: 8)")
    ap.add_argument("--predict", nargs=5, type=int, metavar=("IC", "OC", "IB", "WB", "DSP"),
                    help="Predict LC/FF for IC OC IB WB DSP")
    args = ap.parse_args()

    if args.fill:
        ib, dsp = args.fill
        fill_corner(ib=ib, dsp=dsp, samples=args.samples)

    if args.predict:
        ic, oc, ib, wb, dsp = args.predict
        try:
            lc, ff = predict_conv_layer(ic=ic, oc=oc, ib=ib, wb=wb, dsp=dsp)
            print(f"predict(ic={ic}, oc={oc}, ib={ib}, wb={wb}, dsp={dsp})  =>  LC={lc:.0f}  FF={ff:.0f}")
        except (ValueError, KeyError) as e:
            print(f"predict(ic={ic}, oc={oc}, ib={ib}, wb={wb}, dsp={dsp})  =>  {e}")
    else:
        _report()
