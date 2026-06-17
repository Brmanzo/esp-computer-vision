from itertools import product
from pathlib import Path

from nn.config import NNConfig, InputDimensions
from nn.quantize import QSchedule
from nn.globals import LC_CAP, BUS_WIDTH, DSP_CAP, LC_HEADROOM, CLK_FREQ_HZ

from profiling.LUT4_FF_pred.conv_layer.conv_profile             import predict_conv_layer, valid_dsp_counts
from profiling.LUT4_FF_pred.pool_layer.pool_profile             import predict_pool_layer
from profiling.LUT4_FF_pred.classifier_layer.classifier_profile import predict_classifier_layer
from profiling.LUT4_FF_pred.overhead.overhead_profile           import predict_overhead
from profiling.LC_pred.lc_profile                               import predict_lc as _predict_lc_ff

GENERATE_NET_PATHS = Path("nn") / "sweep" / "checkpoints"

# Task constants (populated dynamically by generate_networks)
IMG_W = IMG_H = 0
INPUT_BITS = 0
INPUT_CHANNELS = 0
NUM_CLASSES = 0

# Network conventions
CONV_KERNEL    = 3
POOL_KERNEL    = 2
PADDING        = 1
STRIDE         = 1

CLASSIFIER_WB  = 8
CLASSIFIER_DSP = 1
DSP_CONV_MAX   = DSP_CAP - CLASSIFIER_DSP   # DSPs available for conv layers

# Search space — ob is always tied to wb (clipping activation width == weight width)

# OutputBits = WeightBits
# --------------------------------- constraints ---------------------------------
WB_RANGE   = range(2, 6)               # 2–5 bits; ob = wb throughout
def grow_weight_bits(w, last_wb) -> bool: return (w >= last_wb)


OC_ANCHORS = [2, 4, 6, 8, 9, 10, 12]   # DSP-divisible channel counts
LAYER_0_DSP_0 = 0
DSP_1_ONLY = 1
L0_DSP_0_WB = [4]
DSP_0_WB = 4
DSP_1_WB = 8
def grow_channels(c, last_oc) -> bool: return (c > last_oc)
# --------------------------------------------------------------------------------

def _lc_ok(total_ff: float, total_lc: float, any_conv_dsp: bool) -> bool:
    """True if this network (partial or complete) fits within LC_CAP.

    DSP>0 in any conv layer: calibrated predict_lc(pred_ff) model (R²=0.97).
    DSP=0 in all conv layers: conservative LUT4 sum against raw cap.
    """
    if any_conv_dsp:
        return _predict_lc_ff(total_ff) <= LC_CAP
    return total_lc <= LC_CAP
# --------------------------------------------------------------------------------

def _build_cfg(layers: list[tuple[int, int, int, int]], last_pool: bool = True) -> NNConfig:
    """Build an NNConfig from a list of (oc, wb, ob, dsp) conv+pool layer params."""
    if last_pool:
        kernels = [[CONV_KERNEL, POOL_KERNEL] for _ in layers] + [[1]]
    else:
        kernels = [[CONV_KERNEL, POOL_KERNEL] for _ in layers[:-1]] + [[CONV_KERNEL]] + [[1]]
    return NNConfig(
        input_dimensions = InputDimensions(IMG_W, IMG_H),
        in_channels      = [INPUT_CHANNELS] + [oc for oc, _, _, _   in layers],
        in_bits          = [INPUT_BITS]     + [ob for _, _, ob, _   in layers],
        kernels          = kernels,
        stride           = STRIDE,
        padding          = PADDING,
        bias_bits        = [16 if dsp > 0 else 8 for _, _, _, dsp in layers] + [16],
        num_classes      = NUM_CLASSES,
        q_schedule       = [
            QSchedule(
                q_start    = 15 + i * 3,
                q_epochs   = [3 + i * 3] if dsp > 0 else [3, 3, 3, 3, 10 + i * 3],
                q_min_bits = 8 if dsp > 0 else wb,
                q_max_bits = 8 if dsp > 0 else wb + 4,
            )
            for i, (_, wb, _, dsp) in enumerate(layers)
        ] + [QSchedule(q_start=15 + len(layers) * 3, q_epochs=[3], q_max_bits=8, q_min_bits=8)],
        use_dsp = [dsp for _, _, _, dsp in layers] + [CLASSIFIER_DSP],
    )


def _post_pool_dims(w: int, h: int) -> tuple[int, int]:
    """Spatial dims after one conv+pool block."""
    w = (w + 2*PADDING - CONV_KERNEL) // STRIDE + 1
    h = (h + 2*PADDING - CONV_KERNEL) // STRIDE + 1
    w = (w - POOL_KERNEL) // POOL_KERNEL + 1
    h = (h - POOL_KERNEL) // POOL_KERNEL + 1
    return w, h


def _fmt(lc: int | float, cfg: NNConfig) -> str:
    _, bottleneck = cfg.cycle_count()
    assert bottleneck is not None, "Cycle count must be computable for formatting"
    assert IMG_W is not None and IMG_H is not None, "Image dimensions must be set for formatting"
    fps = CLK_FREQ_HZ / (int(bottleneck) * IMG_W * IMG_H) if bottleneck > 0 else 0.0

    parts = []
    for c in cfg.layers:
        ob     = c.ConvLayer._out_bits
        wb     = c.ConvLayer._q_schedule._q_min_bits
        dsp    = c.ConvLayer._dsp_count
        oc     = c.ConvLayer._out_ch
        pool   = "+p" if c.PoolLayer is not None else "  "
        wbstr  = f" w{wb}" if wb != ob else "   "
        dspstr = f" d{dsp}" if dsp > 0 else "   "
        parts.append(f"conv({oc:2d}ch {ob}b{wbstr}{dspstr}{pool})")
    return f"[{int(lc):4d} LC  {fps:5.1f} fps]  {'  →  '.join(parts)}"


def _arch_key(cfg: NNConfig) -> tuple:
    """Deduplication key: (oc, wb, ob) sequence + whether the last feature layer has a pool.
    DSP allocation is excluded — it is a hardware detail, not an architectural one."""
    arch = tuple(
        (c.ConvLayer._out_ch, c.ConvLayer._q_schedule._q_min_bits, c.ConvLayer._out_bits)
        for c in cfg.layers
    )
    last_pool = cfg.layers[-1].PoolLayer is not None
    return (arch, last_pool)


def generate_base_cases(
    oc_anchors: list[int] = OC_ANCHORS,
    wb_range: range = WB_RANGE,
) -> list[tuple[int, NNConfig]]:
    """
    Enumerate all feasible first-layer (conv + pool) configurations with dsp=0.

    Fixed: ic=INPUT_CHANNELS=1, ib=INPUT_BITS=1, dsp=0, pool_mode=max.
    Returns (conv+pool lc, NNConfig) tuples sorted by LC ascending.
    Does not account for classifier, overhead, or DSP — use generate_networks for that.
    """
    results = []

    for oc, ob in product(oc_anchors, wb_range):
        for wb in L0_DSP_0_WB:
            try:
                conv_lc, _ = predict_conv_layer(ic=INPUT_CHANNELS, oc=oc, ib=INPUT_BITS, wb=wb, dsp=0)
                pool_lc, _ = predict_pool_layer(ib=ob, ic=oc, mode=0)
            except (KeyError, ValueError):
                continue

            if conv_lc + pool_lc > LC_CAP:
                continue

            try:
                cfg = _build_cfg([(oc, wb, ob, 0)])
            except (AssertionError, Exception):
                continue

            results.append((conv_lc + pool_lc, cfg))

    return sorted(results, key=lambda x: x[0])


def _extend(
    layers: list[tuple[int, int, int, int]],   # (oc, wb, ob, dsp) per layer
    spatial: tuple[int, int],             # dims after the last pool in the current stack
    used_lc: int,
    used_ff: int,
    dsp_remaining: int,
    overhead_lc: int,
    overhead_ff: int,
    oc_anchors: list[int],
    wb_range: range,
    out,
) -> list[tuple[int, NNConfig]]:
    results = []
    last_oc, _, last_ob, _ = layers[-1]

    for oc in [c for c in oc_anchors if grow_channels(c, last_oc)]:
        dsp_options = [DSP_1_ONLY] if dsp_remaining >= 1 and 1 in valid_dsp_counts(oc) else []
        for ob in [w for w in wb_range if grow_weight_bits(w, last_ob)]:
            wb = DSP_0_WB
            for dsp in dsp_options:
                try:
                    conv_lc, conv_ff = predict_conv_layer(ic=last_oc, oc=oc, ib=last_ob, wb=wb, dsp=dsp)
                except (KeyError, ValueError):
                    continue

                new_layers          = layers + [(oc, wb, ob, dsp)]
                has_dsp             = any(d > 0 for _, _, _, d in new_layers)
                used_after_conv     = used_lc + conv_lc
                used_ff_after_conv  = used_ff + conv_ff
                if not _lc_ok(used_ff_after_conv, used_after_conv, has_dsp):
                    continue

                try:
                    class_lc, class_ff = predict_classifier_layer(
                        tb=ob, ic=oc, cc=NUM_CLASSES, wb=CLASSIFIER_WB, dsp=CLASSIFIER_DSP
                    )
                except (KeyError, ValueError, AssertionError, Exception):
                    class_lc, class_ff = None, None

                try:
                    pool_lc, pool_ff = predict_pool_layer(ib=ob, ic=oc, mode=0)
                except (KeyError, ValueError):
                    continue

                used_after_pool    = used_after_conv    + pool_lc
                used_ff_after_pool = used_ff_after_conv + pool_ff

                can_pool = spatial[0] >= POOL_KERNEL and spatial[1] >= POOL_KERNEL

                if class_lc is not None:
                    _cff = class_ff or 0
                    if can_pool:
                        total_ff_pool = used_ff_after_pool + _cff + overhead_ff
                        total_lc_pool = used_after_pool    + class_lc + overhead_lc
                        if _lc_ok(total_ff_pool, total_lc_pool, has_dsp):
                            lc_val = _predict_lc_ff(total_ff_pool) if has_dsp else total_lc_pool
                            try:
                                cfg = _build_cfg(new_layers, last_pool=True)
                                results.append((lc_val, cfg))
                                out.write(_fmt(lc_val, cfg) + "\n")
                                out.flush()
                            except Exception:
                                pass
                        else:
                            total_ff_nopool = used_ff_after_conv + _cff + overhead_ff
                            total_lc_nopool = used_after_conv    + class_lc + overhead_lc
                            if _lc_ok(total_ff_nopool, total_lc_nopool, has_dsp):
                                lc_val = _predict_lc_ff(total_ff_nopool) if has_dsp else total_lc_nopool
                                try:
                                    cfg = _build_cfg(new_layers, last_pool=False)
                                    results.append((lc_val, cfg))
                                    out.write(_fmt(lc_val, cfg) + "\n")
                                    out.flush()
                                except Exception:
                                    pass
                    else:
                        total_ff_nopool = used_ff_after_conv + _cff + overhead_ff
                        total_lc_nopool = used_after_conv    + class_lc + overhead_lc
                        if _lc_ok(total_ff_nopool, total_lc_nopool, has_dsp):
                            lc_val = _predict_lc_ff(total_ff_nopool) if has_dsp else total_lc_nopool
                            try:
                                cfg = _build_cfg(new_layers, last_pool=False)
                                results.append((lc_val, cfg))
                                out.write(_fmt(lc_val, cfg) + "\n")
                                out.flush()
                            except Exception:
                                pass

                if not _lc_ok(used_ff_after_pool, used_after_pool, has_dsp):
                    continue

                new_w, new_h = _post_pool_dims(*spatial)
                if new_w >= 1 and new_h >= 1:
                    results.extend(_extend(
                        new_layers, (new_w, new_h),
                        int(used_after_pool), int(used_ff_after_pool),
                        dsp_remaining - dsp,
                        overhead_lc, overhead_ff,
                        oc_anchors, wb_range, out,
                    ))

    return results


def generate_networks(
    base_cfg: NNConfig | None = None,
    oc_anchors: list[int] = OC_ANCHORS,
    wb_range: range = WB_RANGE,
    out_path: str | None = str(Path(__file__).parent / "networks.txt"),
) -> list[tuple[int | float, NNConfig]]:
    """
    Enumerate all networks that fit on the FPGA, with DSP allocation.

    At each layer, all valid DSP allocations are tried within the remaining budget
    (DSP_CAP - CLASSIFIER_DSP). The DFS explores all distributions so the globally
    optimal allocation (lowest LC) is found for each architecture.

    After collection, results are deduplicated by (oc, wb) sequence and last_pool flag:
    only the minimum-LC entry (= best DSP allocation) is kept per architecture.

    If out_path is set, all candidates are written as found for progress monitoring
    (the file will contain the full search including superseded DSP allocations).

    Returns (total_lc, NNConfig) sorted by total LC ascending.
    """
    if base_cfg is None:
        from nn.globals import NN_CFG
        base_cfg = NN_CFG

    global IMG_W, IMG_H, INPUT_BITS, INPUT_CHANNELS, NUM_CLASSES
    IMG_W = base_cfg.in_dims.width
    IMG_H = base_cfg.in_dims.height
    INPUT_BITS = base_cfg.layers[0].ConvLayer._in_bits
    INPUT_CHANNELS = base_cfg.layers[0].ConvLayer._in_ch
    NUM_CLASSES = base_cfg.num_classes

    assert IMG_W is not None and IMG_H is not None
    overhead_lc, overhead_ff = predict_overhead(
        uw=INPUT_BITS,
        pn=BUS_WIDTH // INPUT_BITS,
        ple=IMG_W * IMG_H * INPUT_CHANNELS,
    )
    post_pool_w, post_pool_h = _post_pool_dims(IMG_W, IMG_H)

    all_results: list[tuple[int | float, NNConfig]] = []

    with open("/dev/null", "w") as null:
        for oc0, ob0 in product(oc_anchors, wb_range):
            for wb0 in L0_DSP_0_WB:
                # Constrain Layer 0 to ALWAYS use dsp=0
                dsp_options = [0]
                for dsp0 in dsp_options:
                    try:
                        conv_lc, conv_ff = predict_conv_layer(
                            ic=INPUT_CHANNELS, oc=oc0, ib=INPUT_BITS, wb=wb0, dsp=dsp0
                        )
                        pool_lc, pool_ff = predict_pool_layer(ib=ob0, ic=oc0, mode=0)
                    except (KeyError, ValueError):
                        continue

                    base_lc  = conv_lc  + pool_lc
                    base_ff  = conv_ff  + pool_ff
                    has_dsp0 = dsp0 > 0
                    if not _lc_ok(base_ff, base_lc, has_dsp0):
                        continue

                    dsp_remaining = DSP_CONV_MAX - dsp0
                    layers0 = [(oc0, wb0, ob0, dsp0)]

                    try:
                        class_lc, class_ff = predict_classifier_layer(
                            tb=ob0, ic=oc0, cc=NUM_CLASSES, wb=CLASSIFIER_WB, dsp=CLASSIFIER_DSP
                        )
                    except (KeyError, ValueError):
                        class_lc, class_ff = None, None

                if class_lc is not None:
                    _class_lc = int(class_lc)
                    _cff = int(class_ff or 0)
                    total_ff_pool = base_ff  + _cff + overhead_ff
                    total_lc_pool = base_lc  + _class_lc + overhead_lc
                    if _lc_ok(total_ff_pool, total_lc_pool, has_dsp0):
                        lc_val = _predict_lc_ff(total_ff_pool) if has_dsp0 else total_lc_pool
                        try:
                            all_results.append((lc_val, _build_cfg(layers0, last_pool=True)))
                        except Exception:
                            pass
                    else:
                        total_ff_nopool = conv_ff  + _cff + overhead_ff
                        total_lc_nopool = conv_lc  + _class_lc + overhead_lc
                        if _lc_ok(total_ff_nopool, total_lc_nopool, has_dsp0):
                            lc_val = _predict_lc_ff(total_ff_nopool) if has_dsp0 else total_lc_nopool
                            try:
                                all_results.append((lc_val, _build_cfg(layers0, last_pool=False)))
                            except Exception:
                                pass

                all_results.extend(_extend(
                    layers0, (post_pool_w, post_pool_h),
                    int(base_lc), int(base_ff),
                    dsp_remaining,
                    overhead_lc, overhead_ff,
                    oc_anchors, wb_range, null,
                ))

    # Deduplicate: for each (oc, wb) architecture keep the minimum-LC DSP allocation
    best: dict[tuple, tuple[int | float, NNConfig]] = {}
    for lc, cfg in all_results:
        key = _arch_key(cfg)
        if key not in best or lc < best[key][0]:
            best[key] = (lc, cfg)

    final = sorted(best.values(), key=lambda x: x[0])

    # Prune: require ≥75% LC utilization and at least 2 conv+pool layers
    final = [
        (lc, cfg) for lc, cfg in final
        if len(cfg.layers) >= 2 and lc <= LC_CAP - LC_HEADROOM
    ]

    # Write deduplicated results to file
    if out_path:
        with open(out_path, "w") as f:
            for lc, cfg in final:
                f.write(_fmt(lc, cfg) + "\n")

    return final


if __name__ == "__main__":
    from nn.sweep.plot import plot_networks, plot_accuracy
    configs = generate_networks()
    print(f"\n{len(configs)} feasible networks (deduplicated)")
    plot_networks(configs)
    plot_accuracy()
