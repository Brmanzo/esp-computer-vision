"""
pool_profile.py — LC and FF lookup for pool_layer on iCE40 UP5K.

Loads measured synthesis results from profiles/sweep_pool.csv.
Keys are (InBits, InChannels, PoolMode); values are exact (LC, FF) counts.

Usage:
    from pool_profile import predict, feasible
    lc, ff = predict_pool_layer(ib=4, ic=8, mode=0)   # mode 0=max, 1=avg
"""

import csv
from pathlib import Path

LC_CAP = 5280
FF_CAP = 5280

_TABLE: dict[tuple, tuple] = {}   # (ib, ic, mode) -> (lc, ff)


def _load(path: Path = Path(__file__).parent / "profiles/profile_coeffs.csv") -> None:
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["InBits"]), int(row["InChannels"]), int(row["PoolMode"]))
            _TABLE[key] = (int(row["LC"]), int(row["FF"]))


def predict_pool_layer(ib: int, ic: int, mode: int = 0) -> tuple[int, int]:
    """
    Return (LC, FF) for a pool_layer configuration.
    Raises KeyError if the configuration was not in the sweep.
    Raises ValueError if either resource exceeds the iCE40 UP5K cap.
    """
    key = (ib, ic, mode)
    if key not in _TABLE:
        raise KeyError(
            f"No measurement for InBits={ib} InChannels={ic} PoolMode={mode} — "
            f"run sweep_pool.py to generate missing data"
        )
    lc, ff = _TABLE[key]
    errors = []
    if lc > LC_CAP:
        errors.append(f"LC={lc} exceeds cap ({LC_CAP})")
    if ff > FF_CAP:
        errors.append(f"FF={ff} exceeds cap ({FF_CAP})")
    if errors:
        raise ValueError(f"IB={ib} IC={ic} mode={mode}: " + ", ".join(errors))
    return lc, ff


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
        feasible_configs = [(ib, ic) for (ib, ic, m), (lc, ff) in _TABLE.items()
                            if m == mode and lc <= LC_CAP]
        print(f"  mode={mode_names[mode]}: {len(feasible_configs)} feasible configs")

    print()
    for mode in [0, 1]:
        for ib in [2, 4, 8]:
            for ic in [4, 8, 16]:
                try:
                    lc, ff = predict_pool_layer(ib=ib, ic=ic, mode=mode)
                    print(f"  predict_pool_layer(ib={ib}, ic={ic:2d}, mode={mode_names[mode]})  =>  LC={lc}  FF={ff}")
                except (KeyError, ValueError) as e:
                    print(f"  predict_pool_layer(ib={ib}, ic={ic:2d}, mode={mode_names[mode]})  =>  {e}")
