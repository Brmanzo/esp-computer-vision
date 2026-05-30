#!/usr/bin/env python3
"""
Convert stat-sweep CSV output to a MATLAB .m column-vector file.

Usage:
    make stat-sweep MOD=... SWEEP=... PARAMS=... PARSE_FLAGS="--fold" | python3 nn/csv_to_matlab.py
    python3 nn/csv_to_matlab.py sweep.csv
    python3 nn/csv_to_matlab.py sweep.csv > sweep.m

Reads labelled CSV produced by stat-sweep (--fold --csv --label) and emits
MATLAB column-vector assignments for the sweep variable, FF, and LUT4 totals.
Each repeated header row is silently skipped.
"""
import sys
import csv
from collections import OrderedDict


def main() -> None:
    src = open(sys.argv[1]) if len(sys.argv) > 1 else sys.stdin
    with src:
        rows = list(csv.reader(src))

    label_key: str | None = None
    # ordered dict: label_val → {"FF": str, "LUT4": str}
    data: OrderedDict[str, dict[str, str]] = OrderedDict()

    for row in rows:
        if not row:
            continue
        # Header rows have "module" in column 1
        if row[1] == "module":
            if label_key is None:
                label_key = row[0]
            continue
        if len(row) < 4:
            continue

        label_val = row[0]
        cell_type = row[3]
        total     = row[-1]

        if cell_type not in ("FF", "LUT4"):
            continue

        if label_val not in data:
            data[label_val] = {}
        data[label_val][cell_type] = total

    if not data:
        sys.exit("No FF/LUT4 rows found — pipe stat-sweep output or pass a CSV file.")

    label_key = label_key or "x"
    labels = list(data.keys())

    def vec(vals: list[str]) -> str:
        return "[" + ";".join(vals) + "]"

    print(f"{label_key} = {vec(labels)};")
    print(f"FF = {vec([data[v].get('FF', '0') for v in labels])};")
    print(f"LUT4 = {vec([data[v].get('LUT4', '0') for v in labels])};")


if __name__ == "__main__":
    main()
