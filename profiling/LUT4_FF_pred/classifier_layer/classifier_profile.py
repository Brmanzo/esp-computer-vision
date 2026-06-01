"""
classifier_profile.py — LUT4 and FF predictor for classifier_layer on iCE40 Icebreaker V1.1a.

Loads regression coefficients from profiles/classifier_coeffs.csv.
Models per (TermBits, DSPCount) corner:
    DSP=0:  LUT4 = A*IC*CC*WB + B*IC + C*log2(CC) + D
    DSP>0:  LUT4 = A*IC*CC    + B*IC + C*log2(CC) + D  (WB fixed by hardware)
    FF DSP=0:  FF = B*IC + D           (A=C=0; no CC·WB or log2(CC) dependence)
    FF DSP>0:  FF = A*IC + B*CC + C*DSP + D  (single model per TB, DSPCount=-1 sentinel in CSV)

Usage:
    from classifier_profile import predict, feasible
    lut4, ff = predict_classifier_layer(tb=4, ic=8, cc=4, wb=4)
    lut4, ff = predict_classifier_layer(tb=4, ic=8, cc=4, wb=4, dsp=2)
"""

import csv
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Tuple

from nn.globals import LC_CAP, DSP_CAP

_LUT4: dict[tuple, tuple] = {}   # (tb, dsp) -> (A, B, C, D, R2)
_FF: dict[tuple, tuple] = {}   # (tb, dsp) -> (A, B, C, D, R2)


def _load(path: Path = Path(__file__).parent / "profiles/classifier_coeffs.csv") -> None:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dsp = int(row["DSPCount"]) if "DSPCount" in row else 0
            key = (int(row["TermBits"]), dsp)
            coeffs = (
                float(row["A"]), float(row["B"]),
                float(row["C"]), float(row["D"]), float(row["R2"]),
            )
            if row["Model"] == "LUT4":
                _LUT4[key] = coeffs
            elif row["Model"] == "FF":
                _FF[key] = coeffs


def _ff(tb: int, ic: int, cc: int, wb: int, dsp: int) -> float:
    key = (tb, 0) if dsp == 0 else (tb, -1)
    if key not in _FF:
        raise KeyError(
            f"No FF model for TermBits={tb} DSPCount={dsp} — "
            f"add FF rows to classifier_coeffs.csv and rerun regression"
        )
    A, B, C, D, _ = _FF[key]
    if dsp == 0:
        return A*ic*cc*wb + B*ic + C*math.log2(max(cc, 1)) + D  # simplifies to B*ic + D (A=C=0)
    else:
        return A*ic + B*cc + C*dsp + D


def _lc(tb: int, ic: int, cc: int, wb: int, dsp: int) -> float:
    key = (tb, dsp)
    if key not in _LUT4:
        if dsp > 0:
            raise KeyError(
                f"No LUT4 model for TermBits={tb} DSPCount={dsp} — "
                f"run fill_corner(tb={tb}, dsp={dsp}) to synthesize it"
            )
        raise KeyError(f"No LUT4 model for TermBits={tb}")
    A, B, C, D, _ = _LUT4[key]
    if dsp == 0:
        return A*ic*cc*wb + B*ic + C*math.log2(max(cc, 1)) + D
    else:
        return A*ic*cc + B*ic + C*math.log2(max(cc, 1)) + D


def predict_classifier_layer(tb: int, ic: int, cc: int, wb: int, dsp: int = 0) -> Tuple[int, int]:
    """
    Predict_classifier_layer (LUT4, FF) for a classifier_layer configuration.
    Raises KeyError if the (TermBits, DSPCount) corner is missing.
    Raises ValueError if cc % dsp != 0 or either resource exceeds the iCE40 Icebreaker V1.1a cap.
    """
    if dsp > 0 and cc % dsp != 0:
        raise ValueError(f"DSPCount={dsp} does not divide ClassCount={cc}")

    try:
        lut4 = _lc(tb, ic, cc, wb, dsp)
    except KeyError:
        if dsp == 0:
            raise
        fill_corner(tb=tb, dsp=dsp)
        lut4 = _lc(tb, ic, cc, wb, dsp)

    try:
        ff = _ff(tb, ic, cc, wb, dsp)
    except KeyError:
        ff = 0

    errors = []
    if lut4 > LC_CAP:
        errors.append(f"LUT4={lut4:.0f} exceeds cap ({LC_CAP})")
    if ff > LC_CAP:
        errors.append(f"FF={ff:.0f} exceeds cap ({LC_CAP})")
    if errors:
        raise ValueError(f"TB={tb} IC={ic} CC={cc} WB={wb} DSP={dsp}: " + ", ".join(errors))
    return math.ceil(lut4), math.ceil(ff)


def feasible(tb: int, ic: int, cc: int, wb: int, dsp: int = 0) -> bool:
    """Return True if the configuration fits within the LUT4 cap."""
    try:
        predict_classifier_layer(tb, ic, cc, wb, dsp)
        return True
    except (KeyError, ValueError):
        return False


def valid_dsp_counts(cc: int) -> list[int]:
    """Divisors of cc that are <= DSP_CAP."""
    return [d for d in range(1, min(cc, DSP_CAP) + 1) if cc % d == 0]


def _count_csv_rows(path: Path, tb: int, dsp: int) -> int:
    if not path.exists():
        return 0
    with open(path, newline="") as f:
        return sum(1 for row in csv.DictReader(f)
                   if int(row["TermBits"]) == tb and int(row["DSPCount"]) == dsp)


def fill_corner(tb: int, dsp: int,
                base_csv: str | None = None,
                samples: int = 8,
                yosys: str = "yosys") -> None:
    """
    Synthesize a missing (tb, dsp) LUT4 corner, refit the regression, and reload.

    Appends new synthesis rows to profiles/classifier_dsp_chars.csv, reruns the
    MATLAB regression to overwrite profiles/classifier_coeffs.csv, then reloads
    coefficients into this process.

    base_csv: path to DSP=0 sweep CSV used for sampling (IC, CC) points.
              Defaults to profiles/sweep_classifier.csv.
    """
    import subprocess

    here    = Path(__file__).parent
    repo    = here.parent.parent
    sweep   = here / "sweep/sweep_classifier_dsp_1_8.py"
    regr    = here / "regression/classifier_dsp_1_8.m"
    dsp_csv = here / "profiles/classifier_dsp_chars.csv"

    if not regr.exists():
        raise FileNotFoundError(
            f"Regression script not found: {regr}\n"
            f"Create classifier_dsp_1_8.m to enable automatic corner fitting."
        )

    if base_csv is None:
        base_csv = str(here / "profiles/sweep_classifier.csv")

    def _run(label: str, cmd: list[str], cwd: Path = repo) -> str:
        print(f"[fill_corner] {label} ...", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
        if r.returncode != 0:
            print(r.stdout[-3000:])
            print(r.stderr[-3000:])
            raise RuntimeError(f"fill_corner: {label} failed (exit {r.returncode})")
        return r.stdout

    rows_before = _count_csv_rows(dsp_csv, tb=tb, dsp=dsp)
    _run(f"Synthesizing TB={tb} DSP={dsp} ({samples} points)",
         ["python3", str(sweep),
          "--base",    base_csv,
          "--tb",      str(tb),
          "--dsp",     str(dsp),
          "--samples", str(samples),
          "--out",     str(dsp_csv),
          "--append",
          "--yosys",   yosys])
    rows_after = _count_csv_rows(dsp_csv, tb=tb, dsp=dsp)
    print(f"[fill_corner] Synthesized {rows_after - rows_before} new rows "
          f"(total TB={tb} DSP={dsp} rows: {rows_after})", flush=True)
    if rows_after < 4:
        print(f"[fill_corner] WARNING: only {rows_after} rows — need ≥4 for regression fit. "
              f"Try --samples with a larger value or check baseline has CC divisible by {dsp}.")

    out = _run("Refitting regression",
               ["matlab", "-batch", f"run('{regr.resolve()}')"],
               cwd=regr.parent)
    for line in out.splitlines():
        if any(w in line.lower() for w in ("corner", "exported", "written")):
            print(f"[fill_corner] MATLAB: {line.strip()}")

    _LUT4.clear()
    _FF.clear()
    _load()
    key = (tb, dsp)
    if key in _LUT4:
        print(f"[fill_corner] Done — corner {key} loaded successfully")
    else:
        print(f"[fill_corner] Corner {key} still missing after refit.")
        print(f"[fill_corner] Available LUT4 corners: {sorted(_LUT4)}")


_FIXED = {"TermCount": 32, "BusBits": 8, "BiasBits": 8, "ShiftBits": 0}
_MODULE = "classifier_layer"


def _get_synth_sources(repo: Path) -> list[str]:
    r = subprocess.run(
        ["python3", "sim/util/get_filelist.py", "rtl/top/top.json"],
        capture_output=True, text=True, cwd=repo,
    )
    return r.stdout.split()


def _synthesize(repo: Path, sources: list[str],
                tb: int, ic: int, cc: int, wb: int, dsp: int,
                yosys: str = "yosys", verbose: bool = False) -> int | None:
    sys.path.insert(0, str(repo))
    from nn.util import parse, total_cells, _fold, _base  # noqa: E402
    params = {**_FIXED, "DSPCount": dsp, "TermBits": tb,
              "InChannels": ic, "ClassCount": cc, "WeightBits": wb}
    param_cmds = "".join(f"chparam -set {k} {v} {_MODULE}; " for k, v in params.items())
    dsp_flag = "-dsp" if dsp > 0 else ""
    script = (
        f"read_verilog -sv -DSYNTHESIS {' '.join(sources)}; "
        f"{param_cmds}"
        f"synth_ice40 {dsp_flag} -noflatten -top {_MODULE}; stat"
    )
    r = subprocess.run([yosys, "-p", script], capture_output=True, text=True, cwd=repo)
    if verbose:
        print(r.stdout)
        if r.stderr:
            print(r.stderr)
    mods    = parse((r.stdout + r.stderr).splitlines())
    top_key = next((k for k in mods if _base(k) == _MODULE), None)
    if top_key is None:
        return None
    tot = _fold(dict(total_cells(top_key, mods, {})))
    return tot.get("LUT4", 0)


_load()


def _report() -> None:
    print(f"Corners loaded — LUT4: {len(_LUT4)}  FF: {len(_FF)}")
    for tb, dsp in sorted(_LUT4):
        A, B, C, D, r2 = _LUT4[(tb, dsp)]
        if dsp == 0:
            print(f"  LUT4  TB={tb} DSP=0   {A:.3f}·IC·CC·WB + {B:.3f}·IC + {C:.3f}·log2(CC) + {D:.3f}  R²={r2:.3f}")
        else:
            print(f"  LUT4  TB={tb} DSP={dsp}  {A:.3f}·IC·CC    + {B:.3f}·IC + {C:.3f}·log2(CC) + {D:.3f}  R²={r2:.3f}")
    for tb, dsp in sorted(_FF):
        A, B, C, D, r2 = _FF[(tb, dsp)]
        if dsp == 0:
            print(f"  FF  TB={tb} DSP=0    {B:.3f}·IC + {D:.3f}  R²={r2:.3f}")
        else:
            print(f"  FF  TB={tb} DSP>0   {A:.3f}·IC + {B:.3f}·CC + {C:.3f}·DSP + {D:.3f}  R²={r2:.3f}")


if __name__ == "__main__":
    import argparse
    import random
    ap = argparse.ArgumentParser(description="Query or fill classifier_layer LUT4 models")
    ap.add_argument("--tb",      type=int, help="TermBits")
    ap.add_argument("--ic",      type=int, help="InChannels")
    ap.add_argument("--cc",      type=int, help="ClassCount")
    ap.add_argument("--wb",      type=int, help="WeightBits")
    ap.add_argument("--dsp",     type=int, default=0, help="DSPCount (default 0)")
    ap.add_argument("--predict", action="store_true", help="Predict LUT4 for --tb --ic --cc --wb --dsp")
    ap.add_argument("--fill",    action="store_true", help="Synthesize and fit a missing DSP corner for --tb --dsp")
    ap.add_argument("--samples", type=int, default=8,
                    help="Sample points per corner when filling (default: 8)")
    ap.add_argument("--trials",  type=int, metavar="N",
                    help="Synthesize N random valid configs and report prediction accuracy")
    ap.add_argument("--seed",    type=int, default=42, help="RNG seed for --trials (default 42)")
    ap.add_argument("--synth",   action="store_true",
                    help="Run synthesis for --tb --ic --cc --wb --dsp and dump yosys log")
    ap.add_argument("--yosys",   default="yosys")
    args = ap.parse_args()

    if args.synth:
        if any(v is None for v in (args.tb, args.ic, args.cc, args.wb)):
            ap.error("--synth requires --tb --ic --cc --wb (and optionally --dsp)")
        repo    = Path(__file__).parent.parent.parent
        sources = _get_synth_sources(repo)
        actual  = _synthesize(repo, sources, args.tb, args.ic, args.cc, args.wb, args.dsp,
                               args.yosys, verbose=True)
        print(f"\nActual LUT4={actual}")
        if actual is not None:
            try:
                pred, _ = predict_classifier_layer(tb=args.tb, ic=args.ic, cc=args.cc, wb=args.wb, dsp=args.dsp)
                print(f"Predicted LUT4={pred}  err={pred - actual:+d}")
            except (KeyError, ValueError) as e:
                print(f"Prediction failed: {e}")

    if args.fill:
        if args.tb is None or args.dsp is None:
            ap.error("--fill requires --tb and --dsp")
        fill_corner(tb=args.tb, dsp=args.dsp, samples=args.samples)

    if args.predict:
        if any(v is None for v in (args.tb, args.ic, args.cc, args.wb)):
            ap.error("--predict requires --tb --ic --cc --wb (and optionally --dsp)")
        tb, ic, cc, wb, dsp = args.tb, args.ic, args.cc, args.wb, args.dsp
        try:
            lut4 = predict_classifier_layer(tb=tb, ic=ic, cc=cc, wb=wb, dsp=dsp)
            print(f"predict(tb={tb}, ic={ic}, cc={cc}, wb={wb}, dsp={dsp})  =>  LUT4={lut4}")
        except (ValueError, KeyError) as e:
            print(f"predict_classifier_layer(tb={tb}, ic={ic}, cc={cc}, wb={wb}, dsp={dsp})  =>  {e}")
    elif args.trials:
        repo    = Path(__file__).parent.parent.parent
        sources = _get_synth_sources(repo)
        if not sources:
            print("ERROR: no source files found — run from repo root", file=sys.stderr)
            sys.exit(1)

        rng = random.Random(args.seed)
        tb_pool = [tb for tb, dsp in _LUT4 if dsp == args.dsp]
        if not tb_pool:
            print(f"ERROR: no corners loaded for DSP={args.dsp}", file=sys.stderr)
            sys.exit(1)

        abs_errors, pct_errors = [], []
        n = 0
        while n < args.trials:
            tb = rng.choice(tb_pool)
            wb = rng.randint(2, 8)
            ic = rng.randint(1, 16)
            cc = rng.randint(1, 16)
            if args.dsp > 0 and cc % args.dsp != 0:
                continue
            try:
                pred, _ = predict_classifier_layer(tb=tb, ic=ic, cc=cc, wb=wb, dsp=args.dsp)
            except (KeyError, ValueError):
                continue
            actual = _synthesize(repo, sources, tb, ic, cc, wb, args.dsp, args.yosys)
            if actual is None:
                continue
            n += 1
            ae = abs(pred - actual)
            pe = 100 * ae / actual if actual else 0
            abs_errors.append(ae)
            pct_errors.append(pe)
            print(f"  [{n:3d}] TB={tb} IC={ic:2d} CC={cc:2d} WB={wb} DSP={args.dsp}"
                  f"  pred={pred:5d}  actual={actual:5d}  err={pred-actual:+d} ({pe:.1f}%)")

        if abs_errors:
            print(f"\nTrials : {n}")
            print(f"Mean |err| : {sum(abs_errors)/n:.1f} LUT4  ({sum(pct_errors)/n:.1f}%)")
            print(f"Max  |err| : {max(abs_errors)} LUT4  ({max(pct_errors):.1f}%)")
            print(f"Std        : {(sum((e - sum(abs_errors)/n)**2 for e in abs_errors)/n)**0.5:.1f} LUT4")
    elif not args.fill:
        _report()
