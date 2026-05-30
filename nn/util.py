#!/usr/bin/env python3
"""
Parse Yosys stat output and print per-module totals that include all
recursive submodule contributions.

Usage:
    make stat-modules | python3 sim/util/stat_parse.py [--top MOD ...]

--top filters by base module name (part after the last backslash in the
$paramod string).  Without --top all modules with iCE40 cells are shown.
"""
import math
import os
import subprocess
import sys
import re
import argparse
from collections import defaultdict

from nn.globals import NN_CFG

_ICE40 = {
    "SB_LUT4", "SB_CARRY",
    "SB_DFF", "SB_DFFE", "SB_DFFR", "SB_DFFS", "SB_DFFSR",
    "SB_DFFER", "SB_DFFESR", "SB_DFFSS",
    "SB_RAM40_4K", "SB_DSP16",
}
_FF_TYPES = {
    "SB_DFF", "SB_DFFE", "SB_DFFR", "SB_DFFS", "SB_DFFSR",
    "SB_DFFER", "SB_DFFESR", "SB_DFFSS",
}


def _fold(d: dict) -> dict:
    """Fold DFF subtypes → 'FF' and collapse SB_LUT4/SB_CARRY → 'LUT4' ceiling."""
    result: defaultdict = defaultdict(int)
    for t, n in d.items():
        if t in _FF_TYPES:
            result["FF"] += n
        elif t not in ("SB_LUT4", "SB_CARRY"):
            result[t] += n
    result["LUT4"] = max(d.get("SB_LUT4", 0), d.get("SB_CARRY", 0))
    return result

def _base(name: str) -> str:
    if "\\" not in name:
        return name
    parts = name.split("\\")
    # Old-style: $paramod\module\param=val\... → module name is parts[1]
    # Hash-style: $paramod$HASH\module       → module name is parts[-1]
    if name.startswith("$paramod\\"):
        return parts[1]
    return parts[-1]


def parse(lines: list[str]) -> dict:
    """
    Returns:
        {full_module_name: {"cells": {type: count},
                            "submodules": {full_name: instance_count}}}

    Only parses lines after the "Printing statistics." marker so synthesis
    log noise (which also uses === ... === headers) is ignored.
    """
    mods: dict = {}
    cur = None
    in_sub = in_cells = started = False

    for raw in lines:
        ln = raw.rstrip()

        # Reset on every new stat section so we always keep the last one
        # (synth_ice40 emits an intermediate stat before tech-mapping;
        # our explicit stat at the end has the real iCE40 primitive counts).
        if "Printing statistics." in ln:
            started = True
            mods = {}
            cur = None
            in_sub = in_cells = False
            continue

        if not started:
            continue

        # New module header
        m = re.match(r"\s*===\s+(.+?)\s*(?:\(partially selected\))?\s*===", ln)
        if m:
            cur = m.group(1).strip()
            mods[cur] = {"cells": defaultdict(int), "submodules": defaultdict(int)}
            in_sub = in_cells = False
            continue

        if cur is None:
            continue

        # Blank line ends the current sub/cell block
        if not ln.strip():
            in_sub = in_cells = False
            continue

        # Stat-header decoration lines ("+------" or bare "|")
        stripped = ln.strip()
        if stripped.startswith("+") or stripped == "|":
            continue

        # "   N cells"      (Local Count / -noflatten format)
        # "Number of cells: N"  (standard stat format)
        if re.match(r"\s+\d+\s+cells\s*$", ln) or re.search(r"Number of cells:", ln):
            in_sub = False
            in_cells = True
            continue

        # "   N submodules" → switch to submodule list mode
        if re.match(r"\s+\d+\s+submodules\s*$", ln):
            in_sub = True
            in_cells = False
            continue

        # Both submodule lines and cell lines share the format:  "   N   name"
        if in_sub:
            m = re.match(r"\s+(\d+)\s+(\S+)", ln)
            if m:
                mods[cur]["submodules"][m.group(2)] += int(m.group(1))

        elif in_cells:
            m = re.match(r"\s+(\d+)\s+(\w+)\s*$", ln)
            if m:
                mods[cur]["cells"][m.group(2)] += int(m.group(1))

    return mods


def total_cells(name: str, mods: dict, memo: dict | None = None) -> defaultdict:
    """Recursively sum local cells + (instance_count x submodule totals)."""
    if memo is None:
        memo = {}
    if name in memo:
        return memo[name]
    if name not in mods:
        memo[name] = defaultdict(int)
        return memo[name]

    result: defaultdict = defaultdict(int)
    for t, n in mods[name]["cells"].items():
        result[t] += n
    for sub, n in mods[name]["submodules"].items():
        for t, cnt in total_cells(sub, mods, memo).items():
            result[t] += cnt * n

    memo[name] = result
    return result


def _print_layer_config(bname: str, layer_idx: int) -> None:
    """Print NN_CFG parameters for one specific layer variant."""
    try:
        from nn.globals import NN_CFG
    except ImportError:
        return
    config = NN_CFG

    if "conv" in bname and "pool" not in bname and "class" not in bname:
        if layer_idx >= len(config.layers):
            return
        c = config.layers[layer_idx].ConvLayer
        cols = ["InBits", "OutBits", "InCh", "OutCh", "KW"]
        vals = [str(c._in_bits), str(c._out_bits), str(c._in_ch), str(c._out_ch), str(c._kernel_width)]
    elif "pool" in bname:
        pool_layers = [(i, lc.PoolLayer) for i, lc in enumerate(config.layers) if lc.PoolLayer]
        if layer_idx >= len(pool_layers):
            return
        _, p = pool_layers[layer_idx]
        cols = ["InBits", "OutBits", "InCh", "OutCh", "KW"]
        vals = [str(p._in_bits), str(p._out_bits), str(p._in_ch), str(p._out_ch), str(p._kernel_width)]
    elif "class" in bname:
        c = config.classifier_config
        cols = ["InBits", "OutBits", "InCh", "Classes"]
        vals = [str(c._in_bits), str(c._out_bits), str(c._in_ch), str(c._num_classes)]
    else:
        return

    widths = [max(len(h), len(v)) for h, v in zip(cols, vals)]
    print("  " + "  ".join(f"{h:>{w}}" for h, w in zip(cols, widths)))
    print("  " + "  ".join("─" * w for w in widths))
    print("  " + "  ".join(f"{v:>{w}}" for v, w in zip(vals, widths)))
    print()


def _abbrev(name: str, max_len: int = 10) -> str:
    """Shorten a module base name to fit in a table column header."""
    if len(name) <= max_len:
        return name
    parts = name.split("_")
    # e.g. multi_delay_buffer → mul_del_buf
    short = "_".join(p[:3] for p in parts)
    return short if len(short) <= max_len else name[:max_len]


def _yosys_stat_module(module: str, params: dict,
                        synth_sources: list[str], yosys: str = "yosys") -> str:
    """Synthesize one module in isolation and return combined stdout+stderr."""
    param_cmds = "".join(f"chparam -set {k} {v} {module}; " for k, v in params.items())
    script = (
        f"read_verilog -sv -DSYNTHESIS {' '.join(synth_sources)}; "
        f"{param_cmds}"
        f"synth_ice40 -dsp -noflatten -top {module}; stat"
    )
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([yosys, "-p", script], capture_output=True, text=True, cwd=repo_root)
    return r.stdout + r.stderr


def design_stat(args: argparse.Namespace) -> None:
    """Synthesize each NN_CFG layer independently and report aggregated totals."""
    synth_sources = os.environ.get("SYNTH_SOURCES", "").split()
    if not synth_sources:
        sys.exit("SYNTH_SOURCES not set — run via `make stat-design`")
    yosys = os.environ.get("YOSYS", "yosys")
    config = NN_CFG
    assert config.in_dims.width is not None and config.in_dims.height is not None
    w = int(config.in_dims.width)
    h = int(config.in_dims.height)
    jobs: list[tuple[str, str, dict]] = []

    for i, layer_cfg in enumerate(config.layers):
        c = layer_cfg.ConvLayer
        conv_params: dict = {
            "LineWidthPx": w, "LineCountPx": h,
            "InBits": c._in_bits, "OutBits": c._out_bits,
            "KernelWidth": c._kernel_width,
            "WeightBits": c._q_schedule._q_min_bits, "BiasBits": c._bias_bits,
            "InChannels": c._in_ch, "OutChannels": c._out_ch,
            "ShiftBits": c._shift, "DSPCount": c._dsp_count,
            "Stride": c._stride, "Padding": c._padding,
        }
        if c._dsp_count == 0:
            # Zero weights get optimized away by Yosys; use a non-zero pattern
            # for accurate LUT-based multiplier area estimation.
            w_bits = c._out_ch * c._in_ch * c._kernel_width * c._kernel_width * c._q_schedule._q_min_bits
            b_bits = c._out_ch * c._bias_bits
            conv_params["Weights"] = f"{w_bits}'h{'5' * ((w_bits + 3) // 4)}"
            conv_params["Biases"]  = f"{b_bits}'h{'5' * ((b_bits + 3) // 4)}"
        jobs.append((f"conv_layer[{i}]", "conv_layer", conv_params))
        p = layer_cfg.PoolLayer
        if p is not None:
            jobs.append((f"pool_layer[{i}]", "pool_layer", {
                "LineWidthPx": w, "LineCountPx": h,
                "InBits": p._in_bits, "OutBits": p._out_bits,
                "KernelWidth": p._kernel_width, "InChannels": p._in_ch,
            }))
            w //= p._kernel_width
            h //= p._kernel_width

    cc = config.classifier_config
    jobs.append(("classifier", "classifier_layer", {
        "TermBits": cc._in_bits, "TermCount": cc._in_ch,
        "BusBits": config._bus_width,
        "InChannels": cc._in_ch, "ClassCount": cc._num_classes,
        "WeightBits": cc._q_schedule._q_min_bits, "BiasBits": cc._bias_bits,
        "ShiftBits": cc._shift, "DSPCount": cc._dsp_count,
    }))

    layer_results: list[tuple[str, dict]] = []
    grand: defaultdict = defaultdict(int)

    for label, module, params in jobs:
        print(f"  [{label}] synthesizing...", file=sys.stderr)
        raw  = _yosys_stat_module(module, params, synth_sources, yosys)
        mods = parse(raw.splitlines())
        if not mods:
            print(f"  Warning: no stat output for {label}. Yosys stderr:", file=sys.stderr)
            for line in raw.splitlines():
                if line.strip():
                    print(f"    {line}", file=sys.stderr)
            continue
        top_key = next((k for k in mods if _base(k) == module), None)
        if top_key is None:
            print(f"  Warning: {module} not in stat output for {label}", file=sys.stderr)
            continue
        tot = total_cells(top_key, mods, {})
        if args.fold:
            tot = _fold(dict(tot))
        layer_results.append((label, tot))
        for t, n in tot.items():
            grand[t] += n

    if args.csv:
        import csv as _csv, io
        all_types = sorted({t for _, tot in layer_results for t in tot})
        rows = [["layer"] + all_types]
        for label, tot in layer_results:
            rows.append([label] + [str(tot.get(t, 0)) for t in all_types])
        rows.append(["TOTAL"] + [str(grand.get(t, 0)) for t in all_types])
        buf = io.StringIO()
        _csv.writer(buf).writerows(rows)
        print(buf.getvalue(), end="")
    else:
        col_w = max(len(lbl) for lbl, _ in layer_results)
        for label, tot in layer_results:
            ram = tot.get("SB_RAM40_4K", 0)
            dsp = tot.get("SB_MAC16", 0)
            if args.fold:
                lc  = tot.get("LUT4", 0)
                ffs = tot.get("FF", 0)
                print(f"  {label:<{col_w}}  LUT4={lc:>5}  FF={ffs:>5}  RAM={ram}  DSP={dsp}")
            else:
                luts  = tot.get("SB_LUT4", 0)
                ffs   = sum(tot.get(t, 0) for t in _FF_TYPES)
                carry = tot.get("SB_CARRY", 0)
                print(f"  {label:<{col_w}}  LUT4={luts:>5}  FF={ffs:>5}  CARRY={carry:>4}  RAM={ram}  DSP={dsp}")
        print(f"\n{'═'*62}")
        ram = grand.get("SB_RAM40_4K", 0)
        dsp = grand.get("SB_MAC16", 0)
        if args.fold:
            print(f"  TOTAL  LUT4={grand.get('LUT4',0)}  FF={grand.get('FF',0)}  RAM={ram}  DSP={dsp}")
        else:
            luts  = grand.get("SB_LUT4", 0)
            ffs   = sum(grand.get(t, 0) for t in _FF_TYPES)
            carry = grand.get("SB_CARRY", 0)
            print(f"  TOTAL  LUT4={luts}  FF={ffs}  CARRY={carry}  RAM={ram}  DSP={dsp}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", nargs="*", metavar="MOD",
                    help="Base module names to report (default: all with iCE40 cells)")
    ap.add_argument("--csv", action="store_true",
                    help="Emit comma-separated output instead of formatted tables")
    ap.add_argument("--fold", action="store_true",
                    help="Fold DFF subtypes → FF and SB_LUT4/SB_CARRY → LUT4 ceiling")
    ap.add_argument("--label", metavar="KEY=VAL",
                    help="Prepend KEY=VAL as first column of every CSV row (use with --csv)")
    ap.add_argument("--design", action="store_true",
                    help="Synthesize each NN_CFG layer independently and report totals")
    args = ap.parse_args()

    if args.design:
        design_stat(args)
        return

    mods = parse(sys.stdin.read().splitlines())
    if not mods:
        sys.exit("No modules parsed — pipe `make stat-modules` output here.")

    groups: dict[str, list[str]] = defaultdict(list)
    for full in mods:
        groups[_base(full)].append(full)

    if args.top:
        want = {b for b in groups if any(kw in b for kw in args.top)}
    else:
        want = set(groups)

    memo: dict = {}
    reported_fulls: list[str] = []   # track every variant we actually print

    csv_mode = args.csv
    csv_rows: list[list[str]] = []   # accumulated for CSV output

    for bname in sorted(want):
        if bname not in groups:
            continue

        if not csv_mode:
            print(f"\n{'═'*62}")
            print(f"  {bname}  ({len(groups[bname])} variant(s))")
            print(f"{'═'*62}")

        def _layer_sort_key(full: str) -> int:
            sub_counts = mods[full]["submodules"]
            # window count == InCh; use it to order conv/pool variants by layer depth.
            # Falls back to total LUT4 count for other module types.
            win = sum(n for s, n in sub_counts.items() if _base(s) == "window")
            if win:
                return win
            return total_cells(full, mods, memo).get("SB_LUT4", 0)

        ordered = sorted(groups[bname], key=_layer_sort_key)

        for layer_idx, full in enumerate(ordered):
            loc  = mods[full]["cells"]
            tot  = total_cells(full, mods, memo)
            subs = mods[full]["submodules"]

            ice_types = sorted({t for t in list(loc) + list(tot) if t in _ICE40})
            if not ice_types and not args.top:
                continue

            reported_fulls.append(full)

            # --- group submodule instances by base name -----------------
            sub_groups: dict[str, dict] = {}
            for sub_full, inst_count in subs.items():
                base = _base(sub_full)
                if base not in sub_groups:
                    sub_groups[base] = {"count": 0, "cells": defaultdict(int)}
                sub_groups[base]["count"] += inst_count
                for t, cnt in total_cells(sub_full, mods, memo).items():
                    sub_groups[base]["cells"][t] += cnt * inst_count

            sub_order = sorted(sub_groups)

            if args.fold:
                loc = _fold(dict(loc))
                tot = _fold(dict(tot))
                for base in sub_order:
                    sub_groups[base]["cells"] = _fold(sub_groups[base]["cells"])
                _display = (_ICE40 - _FF_TYPES - {"SB_LUT4", "SB_CARRY"}) | {"FF", "LUT4"}
                ice_types = sorted({t for t in list(loc) + list(tot) if t in _display})

            if csv_mode:
                # Header row (written once per variant, reader can deduplicate)
                hdr = ["module", "layer", "cell_type", "local"]
                for base in sub_order:
                    hdr.append(f"{base}_x{sub_groups[base]['count']}")
                hdr.append("total")
                csv_rows.append(hdr)
                for t in ice_types:
                    row = [bname, str(layer_idx), t, str(loc.get(t, 0))]
                    for base in sub_order:
                        row.append(str(sub_groups[base]["cells"].get(t, 0)))
                    row.append(str(tot.get(t, 0)))
                    csv_rows.append(row)
            else:
                # --- build column headers ------------------------------------
                col_hdrs = {
                    base: f"{_abbrev(base)} x {sub_groups[base]['count']}"
                    for base in sub_order
                }

                # --- compute column widths ------------------------------------
                type_w = max(len("Cell"), max((len(t) for t in ice_types), default=4))
                loc_w  = max(len("local"),
                             max((len(str(loc.get(t, 0))) for t in ice_types), default=0))
                sub_ws = {
                    base: max(len(col_hdrs[base]),
                              max((len(str(sub_groups[base]["cells"].get(t, 0)))
                                   for t in ice_types), default=0))
                    for base in sub_order
                }
                tot_w  = max(len("total"),
                             max((len(str(tot.get(t, 0))) for t in ice_types), default=0))

                sep_parts = ["─" * type_w, "─" * loc_w]
                hdr_parts = [f"{'Cell':<{type_w}}", f"{'local':>{loc_w}}"]
                for base in sub_order:
                    w = sub_ws[base]
                    hdr_parts.append(f"{col_hdrs[base]:>{w}}")
                    sep_parts.append("─" * w)
                hdr_parts.append(f"{'total':>{tot_w}}")
                sep_parts.append("─" * tot_w)

                print(f"\n{'─'*62}")
                print(f"  {bname}  [Layer {layer_idx}]")
                print(f"  {full}")
                print(f"{'─'*62}")
                _print_layer_config(bname, layer_idx)
                print("  " + "  ".join(hdr_parts))
                print("  " + "  ".join(sep_parts))

                for t in ice_types:
                    row = [f"{t:<{type_w}}", f"{loc.get(t, 0):>{loc_w}}"]
                    for base in sub_order:
                        val = sub_groups[base]["cells"].get(t, 0)
                        row.append(f"{val:>{sub_ws[base]}}")
                    row.append(f"{tot.get(t, 0):>{tot_w}}")
                    print("  " + "  ".join(row))

                ram = tot.get("SB_RAM40_4K", 0)
                dsp = tot.get("SB_MAC16", 0)
                if args.fold:
                    lc  = tot.get("LUT4", 0)
                    ffs = tot.get("FF", 0)
                    print(f"\n  ► LUT4={lc}  FF={ffs}  RAM={ram}  DSP={dsp}")
                else:
                    luts  = tot.get("SB_LUT4", 0)
                    ffs   = sum(tot.get(t, 0) for t in _FF_TYPES)
                    carry = tot.get("SB_CARRY", 0)
                    print(f"\n  ► LUT4={luts}  FF={ffs}  CARRY={carry}"
                          f"  RAM={ram}  DSP={dsp}")

    # --- top-level design total ------------------------------------------
    top_full = next((f for f in mods if _base(f) == "top"), None)
    if top_full is not None:
        top_tot = total_cells(top_full, mods, memo)
        all_ice = sorted({t for t in top_tot if t in _ICE40})
        if all_ice:
            rep_tot: dict[str, int] = defaultdict(int)
            for f in reported_fulls:
                for t, n in total_cells(f, mods, memo).items():
                    rep_tot[t] += n

            if args.fold:
                top_tot = _fold(dict(top_tot))
                rep_tot = _fold(rep_tot)
                all_ice = sorted({t for t in top_tot if t in (_ICE40 - _FF_TYPES - {"SB_LUT4", "SB_CARRY"}) | {"FF", "LUT4"}})

            if csv_mode:
                csv_rows.append(["module", "layer", "cell_type", "reported", "overhead", "total"])
                for t in all_ice:
                    reported = rep_tot.get(t, 0)
                    total_n  = top_tot.get(t, 0)
                    overhead = total_n - reported
                    csv_rows.append(["top", "total", t, str(reported), str(overhead), str(total_n)])
            else:
                cols   = ["Cell", "Reported layers", "Overhead", "Total"]
                rows   = []
                for t in all_ice:
                    reported = rep_tot.get(t, 0)
                    total_n  = top_tot.get(t, 0)
                    overhead = total_n - reported
                    rows.append([t, str(reported), str(overhead), str(total_n)])

                widths = [max(len(cols[i]), max(len(r[i]) for r in rows))
                          for i in range(len(cols))]
                fmt = lambda row: "  " + "  ".join(f"{v:>{w}}" for v, w in zip(row, widths))

                print(f"\n{'═'*62}")
                print(f"  TOP-LEVEL TOTAL")
                print(f"{'═'*62}")
                print(fmt(cols))
                print("  " + "  ".join("─" * w for w in widths))
                for row in rows:
                    print(fmt(row))

                ram = top_tot.get("SB_RAM40_4K", 0)
                dsp = top_tot.get("SB_MAC16", 0)
                if args.fold:
                    lc  = top_tot.get("LUT4", 0)
                    ffs = top_tot.get("FF", 0)
                    print(f"\n  ► LUT4={lc}  FF={ffs}  RAM={ram}  DSP={dsp}")
                else:
                    luts  = top_tot.get("SB_LUT4", 0)
                    ffs   = sum(top_tot.get(t, 0) for t in _FF_TYPES)
                    carry = top_tot.get("SB_CARRY", 0)
                    print(f"\n  ► LUT4={luts}  FF={ffs}  CARRY={carry}"
                          f"  RAM={ram}  DSP={dsp}")

    if csv_mode:
        import csv as _csv
        import io
        if args.label:
            key, val = args.label.split("=", 1)
            csv_rows = [[key if row[0] == "module" else val] + row for row in csv_rows]
        buf = io.StringIO()
        w = _csv.writer(buf)
        w.writerows(csv_rows)
        print(buf.getvalue(), end="")

if __name__ == "__main__":
    main()
