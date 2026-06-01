"""
pool_profile.py — LUT4 and FF lookup for pool_layer on iCE40 Icebreaker V1.1a.

Loads measured synthesis results from profiles/sweep_pool.csv.
Keys are (InBits, InChannels, PoolMode); values are exact (LUT4, FF) counts.

Usage:
    from pool_profile import predict, feasible
    lut4, ff = predict_pool_layer(ib=4, ic=8, mode=0)   # mode 0=max, 1=avg
"""

import csv
from pathlib import Path

from nn.globals import LC_CAP    # iCE40 caps

_TABLE: dict[tuple, tuple] = {}   # (ib, ic, mode) -> (lut4, ff)


def _load(path: Path = Path(__file__).parent / "profiles/profile_coeffs.csv") -> None:
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["InBits"]), int(row["InChannels"]), int(row["PoolMode"]))
            _TABLE[key] = (int(row["LUT4"]), int(row["FF"]))


def predict_pool_layer(ib: int, ic: int, mode: int = 0) -> tuple[int, int]:
    """
    Return (LUT4, FF) for a pool_layer configuration.
    Raises KeyError if the configuration was not in the sweep.
    Raises ValueError if either resource exceeds the iCE40 Icebreaker V1.1a cap.
    """
    key = (ib, ic, mode)
    if key not in _TABLE:
        raise KeyError(
            f"No measurement for InBits={ib} InChannels={ic} PoolMode={mode} — "
            f"run sweep_pool.py to generate missing data"
        )
    lut4, ff = _TABLE[key]
    errors = []
    if lut4 > LC_CAP:
        errors.append(f"LUT4={lut4} exceeds cap ({LC_CAP})")
    if ff > LC_CAP:
        errors.append(f"FF={ff} exceeds cap ({LC_CAP})")
    if errors:
        raise ValueError(f"IB={ib} IC={ic} mode={mode}: " + ", ".join(errors))
    return lut4, ff


def feasible(ib: int, ic: int, mode: int = 0) -> bool:
    """Return True if the configuration was measured and fits within both caps."""
    try:
        predict_pool_layer(ib, ic, mode)
        return True
    except (KeyError, ValueError):
        return False


_load()


if __name__ == "__main__":
    mode_names = {0: "max", 1: "avg"}
    print(f"Configurations loaded: {len(_TABLE)}")
    for mode in [0, 1]:
        feasible_configs = [(ib, ic) for (ib, ic, m), (lut4, ff) in _TABLE.items()
                            if m == mode and lut4 <= LC_CAP]
        print(f"  mode={mode_names[mode]}: {len(feasible_configs)} feasible configs")

    print()
    for mode in [0, 1]:
        for ib in [2, 4, 8]:
            for ic in [4, 8, 16]:
                try:
                    lut4, ff = predict_pool_layer(ib=ib, ic=ic, mode=mode)
                    print(f"  predict_pool_layer(ib={ib}, ic={ic:2d}, mode={mode_names[mode]})  =>  LUT4={lut4}  FF={ff}")
                except (KeyError, ValueError) as e:
                    print(f"  predict_pool_layer(ib={ib}, ic={ic:2d}, mode={mode_names[mode]})  =>  {e}")
