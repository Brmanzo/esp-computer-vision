
from itertools import product

from nn.config import NNConfig, InputDimensions
from nn.quantize import QSchedule
from nn.globals import LC_CAP, BUS_WIDTH

from profiling.conv_layer.conv_profile             import predict_conv_layer
from profiling.pool_layer.pool_profile             import predict_pool_layer
from profiling.classifier_layer.classifier_profile import predict_classifier_layer
from profiling.overhead.overhead_profile           import predict_overhead

# Design constants for task
IMG_W, IMG_H   = 28, 28
INPUT_BITS     = 1
INPUT_CHANNELS = 1
NUM_CLASSES    = 10

# Network conventions
CONV_KERNEL    = 3
POOL_KERNEL    = 2
PADDING        = 1
STRIDE         = 1

CLASSIFIER_WB  = 8
CLASSIFIER_DSP = 1

# Search space — ob is always tied to wb (clipping activation width == weight width)
OC_ANCHORS = [2, 4, 6, 8, 9, 10, 12]   # DSP-friendly channel counts
WB_RANGE   = range(2, 6)               # 2–5 bits; ob = wb throughout


def _build_cfg(layers: list[tuple[int, int]], last_pool: bool = True) -> NNConfig:
    """Build an NNConfig from a list of (oc, wb) pairs. ob is always equal to wb.
    If last_pool=False, the final feature layer connects directly to the classifier."""
    if last_pool:
        kernels = [[CONV_KERNEL, POOL_KERNEL] for _ in layers] + [[1]]
    else:
        kernels = [[CONV_KERNEL, POOL_KERNEL] for _ in layers[:-1]] + [[CONV_KERNEL]] + [[1]]
    return NNConfig(
        input_dimensions = InputDimensions(IMG_W, IMG_H),
        in_channels      = [INPUT_CHANNELS] + [oc for oc, _  in layers],
        in_bits          = [INPUT_BITS]     + [wb for _,  wb in layers],
        kernels          = kernels,
        stride           = STRIDE,
        padding          = PADDING,
        bias_bits        = 8,
        num_classes      = NUM_CLASSES,
        q_schedule       = [
            QSchedule(q_start=0, q_epochs=[15], q_max_bits=wb, q_min_bits=wb)
            for _, wb in layers
        ] + [QSchedule(q_start=0, q_epochs=[15], q_max_bits=4, q_min_bits=4)],
        use_dsp = [0] * (len(layers) + 1),
    )


def _post_pool_dims(w: int, h: int) -> tuple[int, int]:
    """Spatial dims after one conv+pool block."""
    w = (w + 2*PADDING - CONV_KERNEL) // STRIDE + 1
    h = (h + 2*PADDING - CONV_KERNEL) // STRIDE + 1
    w = (w - POOL_KERNEL) // POOL_KERNEL + 1
    h = (h - POOL_KERNEL) // POOL_KERNEL + 1
    return w, h


def _fmt(lc: int | float, cfg: NNConfig) -> str:
    parts = []
    for c in cfg.layers:
        pool = "+p" if c.PoolLayer is not None else ""
        parts.append(f"conv({c.ConvLayer._out_ch}ch {c.ConvLayer._q_schedule._q_min_bits}b{pool})")
    return f"[{int(lc):4d} LC]  {'  →  '.join(parts)}"


def _record(lc: int | float, cfg: NNConfig,
            results: list, out) -> None:
    results.append((lc, cfg))
    out.write(_fmt(lc, cfg) + "\n")
    out.flush()


def generate_base_cases(
    oc_anchors: list[int] = OC_ANCHORS,
    wb_range: range = WB_RANGE,
) -> list[tuple[int, NNConfig]]:
    """
    Enumerate all feasible first-layer (conv + pool) configurations.

    Fixed: ic=INPUT_CHANNELS=1, ib=INPUT_BITS=1, dsp=0, pool_mode=max, ob=wb.
    Returns (conv+pool lc, NNConfig) tuples sorted by LC ascending.
    Does not account for classifier or overhead — those are added in generate_networks.
    """
    results = []

    for oc, wb in product(oc_anchors, wb_range):
        try:
            conv_lc, _ = predict_conv_layer(ic=INPUT_CHANNELS, oc=oc, ib=INPUT_BITS, wb=wb, dsp=0)
            pool_lc, _ = predict_pool_layer(ib=wb, ic=oc, mode=0)
        except (KeyError, ValueError):
            continue

        if conv_lc + pool_lc > LC_CAP:
            continue

        try:
            cfg = _build_cfg([(oc, wb)])
        except (AssertionError, Exception):
            continue

        results.append((conv_lc + pool_lc, cfg))

    return sorted(results, key=lambda x: x[0])


def _extend(
    layers: list[tuple[int, int]],   # (oc, wb) per layer; oc and wb non-decreasing
    spatial: tuple[int, int],        # dims after the last pool in the current stack
    used_lc: int,
    overhead_lc: int,
    oc_anchors: list[int],
    wb_range: range,
    out,
) -> list[tuple[int, NNConfig]]:
    """
    Recursively extend the current layer stack with one more conv block.

    At each depth two termination paths are tried before recursing:
      1. conv (no final pool) → classifier
      2. conv + pool → classifier
    Recursion only follows path 2 (further layers require a pool).

    Constraints: oc strictly greater than previous, wb >= previous wb, ob = wb.
    Each valid config is written to out immediately.
    """
    results = []
    last_oc, last_wb = layers[-1]

    for oc in [c for c in oc_anchors if c > last_oc]:
        for wb in [w for w in wb_range if w >= last_wb]:
            # Conv is always required
            try:
                conv_lc, _ = predict_conv_layer(ic=last_oc, oc=oc, ib=last_wb, wb=wb, dsp=0)
            except (KeyError, ValueError):
                continue

            used_after_conv = used_lc + conv_lc
            if used_after_conv > LC_CAP:
                continue

            new_layers = layers + [(oc, wb)]

            # Classifier cost is shared by both termination paths
            try:
                class_lc, _ = predict_classifier_layer(
                    tb=wb, ic=oc, cc=NUM_CLASSES, wb=CLASSIFIER_WB, dsp=CLASSIFIER_DSP
                )
            except (KeyError, ValueError, AssertionError, Exception):
                class_lc = None

            # Pool cost — needed for pool termination and recursion
            try:
                pool_lc, _ = predict_pool_layer(ib=wb, ic=oc, mode=0)
            except (KeyError, ValueError):
                continue

            used_after_pool = used_after_conv + pool_lc

            if class_lc is not None:
                total_with_pool = used_after_pool + class_lc + overhead_lc
                if total_with_pool <= LC_CAP:
                    # Path 1: conv + pool → classifier (preferred)
                    try:
                        _record(total_with_pool, _build_cfg(new_layers, last_pool=True), results, out)
                    except Exception:
                        pass
                else:
                    # Path 2: conv (no pool) → classifier, only if pool doesn't fit
                    total_no_pool = used_after_conv + class_lc + overhead_lc
                    if total_no_pool <= LC_CAP:
                        try:
                            _record(total_no_pool, _build_cfg(new_layers, last_pool=False), results, out)
                        except Exception:
                            pass

            if used_after_pool > LC_CAP:
                continue

            # Recurse: add yet another layer after this pool
            new_w, new_h = _post_pool_dims(*spatial)
            if new_w >= 1 and new_h >= 1:
                results.extend(_extend(
                    new_layers, (new_w, new_h), int(used_after_pool), overhead_lc,
                    oc_anchors, wb_range, out,
                ))

    return results


def generate_networks(
    oc_anchors: list[int] = OC_ANCHORS,
    wb_range: range = WB_RANGE,
    out_path: str | None = "nn/tasks/generate/networks.txt",
) -> list[tuple[int, NNConfig]]:
    """
    Enumerate all networks that fit on the FPGA.

    For each base case, tries two 1-layer terminations (with and without final pool),
    then recursively extends with more conv+pool layers. oc and wb are non-decreasing
    across layers; ob = wb throughout.

    If out_path is set, each valid config is written and flushed immediately.
    Returns (total_lc, NNConfig) sorted by total LC ascending.
    """
    overhead_lc, _ = predict_overhead(
        uw=INPUT_BITS,
        pn=BUS_WIDTH // INPUT_BITS,
        ple=IMG_W * IMG_H * INPUT_CHANNELS,
    )
    post_pool_w, post_pool_h = _post_pool_dims(IMG_W, IMG_H)

    results = []
    with (open(out_path, "w") if out_path else open("/dev/null", "w")) as out:
        for base_lc, cfg in generate_base_cases(oc_anchors, wb_range):
            c   = cfg.layers[0].ConvLayer
            oc0 = c._out_ch
            wb0 = c._q_schedule._q_min_bits

            try:
                class_lc, _ = predict_classifier_layer(
                    tb=wb0, ic=oc0, cc=NUM_CLASSES, wb=CLASSIFIER_WB, dsp=CLASSIFIER_DSP
                )
            except (KeyError, ValueError):
                class_lc = None

            if class_lc is not None:
                total_with_pool = base_lc + class_lc + overhead_lc
                if total_with_pool <= LC_CAP:
                    # 1-layer with pool (cfg already built with pool)
                    _record(total_with_pool, cfg, results, out)
                else:
                    # 1-layer without pool — fallback when pool pushes over cap
                    try:
                        conv_lc, _ = predict_conv_layer(
                            ic=INPUT_CHANNELS, oc=oc0, ib=INPUT_BITS, wb=wb0, dsp=0
                        )
                        total_no_pool = conv_lc + class_lc + overhead_lc
                        if total_no_pool <= LC_CAP:
                            _record(total_no_pool, _build_cfg([(oc0, wb0)], last_pool=False),
                                    results, out)
                    except (KeyError, ValueError, Exception):
                        pass

            # Extend: base_lc includes pool cost, correct starting point for recursion
            results.extend(_extend(
                [(oc0, wb0)], (post_pool_w, post_pool_h), base_lc, overhead_lc,
                oc_anchors, wb_range, out,
            ))

    return sorted(results, key=lambda x: x[0])


if __name__ == "__main__":
    configs = generate_networks()
    print(f"\n{len(configs)} feasible networks")
