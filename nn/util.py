#!/usr/bin/env python3
"""
Parse Yosys stat output and print per-module totals that include all
recursive submodule contributions.

Usage:
    make stat-modules | python3 sim/util/stat_parse.py [--top MOD ...]

--top filters by base module name (part after the last backslash in the
$paramod string).  Without --top all modules with iCE40 cells are shown.
"""
import sys
import re
import argparse
from collections import defaultdict

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


def _base(name: str) -> str:
    return name.split("\\")[-1] if "\\" in name else name


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
    """Recursively sum local cells + (instance_count × submodule totals)."""
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", nargs="*", metavar="MOD",
                    help="Base module names to report (default: all with iCE40 cells)")
    args = ap.parse_args()

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

    for bname in sorted(want):
        if bname not in groups:
            continue

        for full in sorted(groups[bname]):
            loc  = mods[full]["cells"]
            tot  = total_cells(full, mods, memo)
            subs = mods[full]["submodules"]

            ice_types = sorted({t for t in list(loc) + list(tot) if t in _ICE40})
            if not ice_types and not args.top:
                continue  # skip pure-wiring / wrapper modules when in default mode

            print(f"\n{'─'*62}")
            print(f"  {bname}")
            print(f"  {full}")
            print(f"{'─'*62}")

            if subs:
                print("  Submodules (direct children):")
                for s, n in sorted(subs.items(), key=lambda x: -x[1]):
                    print(f"    {n:>3}×  {_base(s)}")
                print()

            if ice_types:
                print(f"  {'Cell':<18} {'Local':>8} {'Total (incl. sub)':>18}")
                print(f"  {'─'*18} {'─'*8} {'─'*18}")
                for t in ice_types:
                    print(f"  {t:<18} {loc.get(t, 0):>8} {tot.get(t, 0):>18}")

            luts   = tot.get("SB_LUT4", 0)
            ffs    = sum(tot.get(t, 0) for t in _FF_TYPES)
            carry  = tot.get("SB_CARRY", 0)
            ram    = tot.get("SB_RAM40_4K", 0)
            dsp    = tot.get("SB_MAC16", 0)
            print(f"\n  ► LUT4={luts}  FF={ffs}  CARRY={carry}"
                  f"  RAM={ram}  DSP={dsp}")


if __name__ == "__main__":
    main()
