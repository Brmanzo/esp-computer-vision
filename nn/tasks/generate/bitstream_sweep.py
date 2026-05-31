#!/usr/bin/env python3
"""
nn/tasks/generate/bitstream_sweep.py

For each trained network from the generate sweep:
  1. Export weights       (cnn.py export)
  2. Render SystemVerilog (cnn.py verilog)
  3. make bitstream ESP=0
  4. make util (ICESTORM_LC) + make stat (LUTs, DFFs from icebox_stat)
  5. Log prediction accuracy to bitstream_results.txt

Columns:
  idx  pred_lut4  act_lut4  lut4_err%  pred_ff  act_ff  ff_err%  act_lc  status  arch

Run from repo root:
    python nn/tasks/generate/bitstream_sweep.py
    python nn/tasks/generate/bitstream_sweep.py --start-from 12
"""

import argparse
import os
import re
import signal
import subprocess
from pathlib import Path

from nn.config  import NNConfig
from nn.export  import export_nn_to_csv, export_csv_to_hex
from nn.globals import LC_CAP
from nn.tasks.generate.generate import generate_networks, _fmt
from nn.verilog import render_verilog

from profiling.profile_model import profile_model

REPO_ROOT          = Path(__file__).resolve().parents[3]
DATA_DIR           = Path("nn") / "data"
CKPT_DIR           = Path("nn") / "tasks" / "generate" / "checkpoints"
TRAIN_RESULTS_PATH = Path("nn") / "tasks" / "generate" / "results.txt"
RESULTS_PATH       = Path("nn") / "tasks" / "generate" / "bitstream_results.txt"

_CSV_PATH = DATA_DIR / "hardware_weights.csv"
_VH_PATH  = DATA_DIR / "hardware_weights.vh"
_HEX_DIR  = DATA_DIR / "roms" / "hex"


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_aborted(path: Path) -> set[int]:
    """Return set of network indices marked 'aborted' in results.txt."""
    aborted: set[int] = set()
    if not path.exists():
        return aborted
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit() and parts[1] == "aborted":
                aborted.add(int(parts[0]))
    return aborted




def _parse_util(nplog_text: str) -> int | None:
    """Extract ICESTORM_LC count from nextpnr log."""
    m = re.search(r'ICESTORM_LC:\s+(\d+)/', nplog_text)
    return int(m.group(1)) if m else None


def _parse_stat(text: str) -> tuple[int | None, int | None]:
    """Parse LUTs and DFFs from icebox_stat output (make stat).

    icebox_stat reports post-P&R bitstream counts — ground-truth values
    after nextpnr packs cells into logic tiles.  Format:
        DFFs:    960
        LUTs:   2275
        ...
    """
    lut_m = re.search(r'LUTs:\s+(\d+)', text)
    dff_m = re.search(r'DFFs:\s+(\d+)', text)
    return (int(lut_m.group(1)) if lut_m else None,
            int(dff_m.group(1)) if dff_m else None)


def _err_pct(actual: int | None, pred: float | int) -> str:
    """Format (actual-pred)/actual as a signed percentage string."""
    if actual is None or actual == 0:
        return "    —  "
    return f"{100 * (actual - pred) / actual:+.1f}%"


def _make(*targets: str, timeout: int | None = None, **vars_) -> subprocess.CompletedProcess:
    cmd = ["make"] + list(targets) + [f"{k}={v}" for k, v in vars_.items()]
    if timeout is None:
        return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    # Use a new session so the whole process tree (make + nextpnr) shares a
    # process group — os.killpg then kills everything, not just the make shell.
    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        cwd=REPO_ROOT, start_new_session=True,
    ) as proc:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.communicate()  # drain pipes so no zombie is left
            raise


# ── sweep ─────────────────────────────────────────────────────────────────────

def sweep(start_from: int = 0) -> None:
    configs = generate_networks()
    total   = len(configs)
    aborted = _load_aborted(TRAIN_RESULTS_PATH)
    print(f"\n{total} networks to sweep ({len(aborted)} aborted in results.txt, starting from #{start_from})\n")

    _HEX_DIR.mkdir(parents=True, exist_ok=True)

    HDR = (f"{'idx':>4}  {'pred_lut4':>9}  {'act_lut4':>8}  {'lut4_err':>8}  "
           f"{'pred_ff':>7}  {'act_ff':>6}  {'ff_err':>7}  {'act_lc':>6}  "
           f"{'status':<16}  arch\n")

    file_mode = "a" if start_from > 0 else "w"
    with open(RESULTS_PATH, file_mode, buffering=1) as f:
        if start_from == 0:
            f.write(HDR)
            f.write("─" * 160 + "\n")

        for i, (pred_lc, cfg) in enumerate(configs):
            if i < start_from:
                continue

            header = _fmt(pred_lc, cfg)
            print(f"\n[{i:3d}/{total-1}] {header}", flush=True)

            # ── skip conditions ──────────────────────────────────────────────
            if i in aborted:
                print(f"  [skip] aborted in results.txt", flush=True)
                _write_skip(f, i, cfg, "aborted", header)
                continue

            ckpt = CKPT_DIR / f"network_{i:04d}.pth"
            if not ckpt.exists():
                print(f"  [skip] no checkpoint", flush=True)
                _write_skip(f, i, cfg, "no_ckpt", header)
                continue

            # ── compute predicted LUT4 / FF ──────────────────────────────────
            try:
                pred_lut4, pred_ff = profile_model(cfg)
            except Exception as e:
                print(f"  [warn] prediction failed: {e}", flush=True)
                pred_lut4, pred_ff = int(pred_lc), 0

            if abs(pred_lut4 - pred_lc) > 1:
                print(f"  [warn] profile_model LUT4={pred_lut4} differs from generate pred_lc={int(pred_lc)} "
                      f"(Δ={pred_lut4 - int(pred_lc):+.1f}) — prediction mismatch", flush=True)

            # ── 1. export ────────────────────────────────────────────────────
            print("  → export ...", flush=True)
            try:
                export_nn_to_csv(ckpt, cfg, _CSV_PATH)
                export_csv_to_hex(_CSV_PATH, _VH_PATH, _HEX_DIR, cfg)
            except Exception as e:
                print(f"  [error] export: {e}", flush=True)
                _write_row(f, i, pred_lut4, None, pred_ff, None, None, "export_err", header)
                continue

            # ── 2. verilog ───────────────────────────────────────────────────
            print("  → verilog ...", flush=True)
            try:
                render_verilog(cfg)
            except Exception as e:
                print(f"  [error] verilog: {e}", flush=True)
                _write_row(f, i, pred_lut4, None, pred_ff, None, None, "verilog_err", header)
                continue

            # ── 3. bitstream ─────────────────────────────────────────────────
            print("  → make bitstream ESP=0 ...", flush=True)
            try:
                r = _make("bitstream", timeout=60, ESP=0)
            except subprocess.TimeoutExpired:
                print("  [error] timed out after 60s — nextpnr hung", flush=True)
                _write_row(f, i, pred_lut4, None, pred_ff, None, None, "timeout", header)
                continue
            if r.returncode != 0:
                combined = r.stdout + r.stderr
                lc_exceeded = any(
                    kw in combined.lower()
                    for kw in ("unable to place", "placement aborted",
                               "routing failed", "failed to route", "could not be placed")
                )
                status = "lc_cap_exceeded" if lc_exceeded else "build_err"
                print(f"  [error] {'LC Cap exceeded' if lc_exceeded else 'build error'}", flush=True)
                tail = combined[-600:].strip()
                if tail:
                    print(f"  {tail}", flush=True)
                _write_row(f, i, pred_lut4, None, pred_ff, None, None, status, header)
                continue

            # ── 4. actual counts ─────────────────────────────────────────────
            print("  → make util + stat ...", flush=True)
            r_util = _make("util")
            r_stat = _make("stat")
            actual_lc        = _parse_util(r_util.stdout)
            act_lut4, act_ff = _parse_stat(r_stat.stdout)

            over_cap = actual_lc is not None and actual_lc > LC_CAP
            status   = "lc_cap_exceeded" if over_cap else "ok"

            print(
                f"  → lut4: pred={pred_lut4}  act={act_lut4}  err={_err_pct(act_lut4, pred_lut4).strip()}\n"
                f"     ff:  pred={pred_ff}  act={act_ff}  err={_err_pct(act_ff, pred_ff).strip()}\n"
                f"     lc:  act={actual_lc}"
                + ("  ← EXCEEDS CAP" if over_cap else ""),
                flush=True,
            )
            _write_row(f, i, pred_lut4, act_lut4, pred_ff, act_ff, actual_lc, status, header)

    print(f"\nResults written to {RESULTS_PATH}")


# ── formatting helpers ────────────────────────────────────────────────────────

def _write_skip(f, i: int, cfg: NNConfig, status: str, header: str) -> None:
    try:
        pred_lut4, pred_ff = profile_model(cfg)
    except Exception:
        pred_lut4, pred_ff = 0, 0
    _write_row(f, i, pred_lut4, None, pred_ff, None, None, status, header)


def _write_row(
    f, i: int,
    pred_lut4: float | int, act_lut4: int | None,
    pred_ff: float | int,   act_ff:   int | None,
    act_lc:   int | None,
    status: str, header: str,
) -> None:
    def _fmt_int(v: int | None) -> str:
        return f"{v:>8}" if v is not None else f"{'—':>8}"

    f.write(
        f"{i:4d}  {pred_lut4:>9}  {_fmt_int(act_lut4)}  {_err_pct(act_lut4, pred_lut4):>8}  "
        f"{pred_ff:>7}  {_fmt_int(act_ff):>6}  {_err_pct(act_ff, pred_ff):>7}  "
        f"{_fmt_int(act_lc):>6}  {status:<16}  {header}\n"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Sweep export→verilog→bitstream→LC check for all trained networks")
    ap.add_argument("--start-from", type=int, default=0,
                    help="Resume from this network index (appends to existing results)")
    args = ap.parse_args()
    sweep(start_from=args.start_from)
