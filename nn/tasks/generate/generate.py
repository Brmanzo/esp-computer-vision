
from itertools import product

from nn.config import NNConfig, InputDimensions
from nn.quantize import QSchedule
from nn.globals import LC_CAP, BUS_WIDTH, DSP_CAP, LC_HEADROOM, CLK_FREQ_HZ

from profiling.conv_layer.conv_profile             import predict_conv_layer, valid_dsp_counts
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
DSP_CONV_MAX   = DSP_CAP - CLASSIFIER_DSP   # DSPs available for conv layers

# Search space — ob is always tied to wb (clipping activation width == weight width)
OC_ANCHORS = [2, 4, 6, 8, 9, 10, 12]   # DSP-friendly channel counts
WB_RANGE   = range(2, 6)               # 2–5 bits; ob = wb throughout


def _build_cfg(layers: list[tuple[int, int, int]], last_pool: bool = True) -> NNConfig:
    """Build an NNConfig from a list of (oc, wb, dsp) conv+pool layer params.
    ob = wb always. If last_pool=False the final feature layer feeds the classifier directly."""
    if last_pool:
        kernels = [[CONV_KERNEL, POOL_KERNEL] for _ in layers] + [[1]]
    else:
        kernels = [[CONV_KERNEL, POOL_KERNEL] for _ in layers[:-1]] + [[CONV_KERNEL]] + [[1]]
    return NNConfig(
        input_dimensions = InputDimensions(IMG_W, IMG_H),
        in_channels      = [INPUT_CHANNELS] + [oc for oc, _,  _   in layers],
        in_bits          = [INPUT_BITS]     + [wb for _,  wb, _   in layers],
        kernels          = kernels,
        stride           = STRIDE,
        padding          = PADDING,
        bias_bits        = [16 if dsp > 0 else 8 for _, _, dsp in layers] + [16],
        num_classes      = NUM_CLASSES,
        q_schedule       = [
            QSchedule(
                q_start    = 25 + i * 5,
                q_epochs   = [3, 3, 3, 3, 10 + i * 5],
                q_min_bits = 8 if dsp > 0 else wb,
                q_max_bits = (8 if dsp > 0 else wb) + 4,
            )
            for i, (_, wb, dsp) in enumerate(layers)
        ] + [QSchedule(q_start=0, q_epochs=[15], q_max_bits=4, q_min_bits=4)],
        use_dsp = [dsp for _, _, dsp in layers] + [CLASSIFIER_DSP],
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
    fps = CLK_FREQ_HZ / (bottleneck * IMG_W * IMG_H) if bottleneck > 0 else 0.0

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
    """Deduplication key: (oc, wb) sequence + whether the last feature layer has a pool.
    DSP allocation is excluded — it is a hardware detail, not an architectural one."""
    arch = tuple(
        (c.ConvLayer._out_ch, c.ConvLayer._q_schedule._q_min_bits)
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

    Fixed: ic=INPUT_CHANNELS=1, ib=INPUT_BITS=1, dsp=0, pool_mode=max, ob=wb.
    Returns (conv+pool lc, NNConfig) tuples sorted by LC ascending.
    Does not account for classifier, overhead, or DSP — use generate_networks for that.
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
            cfg = _build_cfg([(oc, wb, 0)])
        except (AssertionError, Exception):
            continue

        results.append((conv_lc + pool_lc, cfg))

    return sorted(results, key=lambda x: x[0])


def _extend(
    layers: list[tuple[int, int, int]],   # (oc, wb, dsp) per layer
    spatial: tuple[int, int],             # dims after the last pool in the current stack
    used_lc: int,
    dsp_remaining: int,
    overhead_lc: int,
    oc_anchors: list[int],
    wb_range: range,
    out,
) -> list[tuple[int, NNConfig]]:
    """
    Recursively extend the current layer stack with one more conv block.

    For each candidate (oc, wb), all valid DSP allocations are tried (descending,
    so lower-LC allocations are found first). At each depth two termination paths
    are tried before recursing:
      1. conv + pool → classifier  (preferred)
      2. conv (no pool) → classifier  (fallback if path 1 exceeds LC_CAP)
    Recursion only follows path 1 (further layers require a pool).

    Constraints: oc strictly greater than previous, wb >= previous wb, ob = wb.
    All valid configs are written to out immediately for progress monitoring.
    Deduplication on (oc, wb) architecture is done in generate_networks after collection.
    """
    results = []
    last_oc, last_wb, _ = layers[-1]

    for oc in [c for c in oc_anchors if c > last_oc]:
        dsp_options = sorted(
            [0] + [d for d in valid_dsp_counts(oc) if d <= dsp_remaining],
            reverse=True,   # max DSP first → lower LC explored first
        )
        for wb in [w for w in wb_range if w >= last_wb]:
            for dsp in dsp_options:
                try:
                    conv_lc, _ = predict_conv_layer(ic=last_oc, oc=oc, ib=last_wb, wb=wb, dsp=dsp)
                except (KeyError, ValueError):
                    continue

                used_after_conv = used_lc + conv_lc
                if used_after_conv > LC_CAP:
                    continue

                new_layers = layers + [(oc, wb, dsp)]

                try:
                    class_lc, _ = predict_classifier_layer(
                        tb=wb, ic=oc, cc=NUM_CLASSES, wb=CLASSIFIER_WB, dsp=CLASSIFIER_DSP
                    )
                except (KeyError, ValueError, AssertionError, Exception):
                    class_lc = None

                try:
                    pool_lc, _ = predict_pool_layer(ib=wb, ic=oc, mode=0)
                except (KeyError, ValueError):
                    continue

                used_after_pool = used_after_conv + pool_lc

                if class_lc is not None:
                    total_with_pool = used_after_pool + class_lc + overhead_lc
                    if total_with_pool <= LC_CAP:
                        try:
                            cfg = _build_cfg(new_layers, last_pool=True)
                            results.append((total_with_pool, cfg))
                            out.write(_fmt(total_with_pool, cfg) + "\n")
                            out.flush()
                        except Exception:
                            pass
                    else:
                        total_no_pool = used_after_conv + class_lc + overhead_lc
                        if total_no_pool <= LC_CAP:
                            try:
                                cfg = _build_cfg(new_layers, last_pool=False)
                                results.append((total_no_pool, cfg))
                                out.write(_fmt(total_no_pool, cfg) + "\n")
                                out.flush()
                            except Exception:
                                pass

                if used_after_pool > LC_CAP:
                    continue

                new_w, new_h = _post_pool_dims(*spatial)
                if new_w >= 1 and new_h >= 1:
                    results.extend(_extend(
                        new_layers, (new_w, new_h), int(used_after_pool),
                        dsp_remaining - dsp, overhead_lc,
                        oc_anchors, wb_range, out,
                    ))

    return results


def generate_networks(
    oc_anchors: list[int] = OC_ANCHORS,
    wb_range: range = WB_RANGE,
    out_path: str | None = "nn/tasks/generate/networks.txt",
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
    overhead_lc, _ = predict_overhead(
        uw=INPUT_BITS,
        pn=BUS_WIDTH // INPUT_BITS,
        ple=IMG_W * IMG_H * INPUT_CHANNELS,
    )
    post_pool_w, post_pool_h = _post_pool_dims(IMG_W, IMG_H)

    all_results: list[tuple[int | float, NNConfig]] = []

    with open("/dev/null", "w") as null:
        for oc0, wb0 in product(oc_anchors, wb_range):
            dsp_options = sorted(
                [0] + [d for d in valid_dsp_counts(oc0) if d <= DSP_CONV_MAX],
                reverse=True,
            )
            for dsp0 in dsp_options:
                try:
                    conv_lc, _ = predict_conv_layer(
                        ic=INPUT_CHANNELS, oc=oc0, ib=INPUT_BITS, wb=wb0, dsp=dsp0
                    )
                    pool_lc, _ = predict_pool_layer(ib=wb0, ic=oc0, mode=0)
                except (KeyError, ValueError):
                    continue

                base_lc = conv_lc + pool_lc
                if base_lc > LC_CAP:
                    continue

                dsp_remaining = DSP_CONV_MAX - dsp0
                layers0 = [(oc0, wb0, dsp0)]

                try:
                    class_lc, _ = predict_classifier_layer(
                        tb=wb0, ic=oc0, cc=NUM_CLASSES, wb=CLASSIFIER_WB, dsp=CLASSIFIER_DSP
                    )
                except (KeyError, ValueError):
                    class_lc = None

                if class_lc is not None:
                    total_with_pool = base_lc + class_lc + overhead_lc
                    if total_with_pool <= LC_CAP:
                        try:
                            all_results.append((total_with_pool, _build_cfg(layers0, last_pool=True)))
                        except Exception:
                            pass
                    else:
                        total_no_pool = conv_lc + class_lc + overhead_lc
                        if total_no_pool <= LC_CAP:
                            try:
                                all_results.append((total_no_pool, _build_cfg(layers0, last_pool=False)))
                            except Exception:
                                pass

                all_results.extend(_extend(
                    layers0, (post_pool_w, post_pool_h), int(base_lc),
                    dsp_remaining, overhead_lc,
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
        if lc >= 0.75 * LC_CAP and len(cfg.layers) >= 2 and lc <= LC_CAP - LC_HEADROOM
    ]

    # Write deduplicated results to file
    if out_path:
        with open(out_path, "w") as f:
            for lc, cfg in final:
                f.write(_fmt(lc, cfg) + "\n")

    return final


def plot_networks(
    configs: list[tuple[int | float, NNConfig]],
    out_html: str = "nn/tasks/generate/networks_plot.html",
) -> None:
    """Scatter plot of LC vs FPS for all enumerated networks.
    Hover over a point to see its index and layer architecture."""
    import plotly.graph_objects as go

    lcs, fpss, hover = [], [], []
    for i, (lc, cfg) in enumerate(configs):
        _, bottleneck = cfg.cycle_count()
        fps = CLK_FREQ_HZ / (bottleneck * IMG_W * IMG_H) if bottleneck > 0 else 0.0

        layer_str = "<br>".join(
            "  " + (
                f"conv({c.ConvLayer._out_ch}ch {c.ConvLayer._out_bits}b"
                + (f" w{c.ConvLayer._q_schedule._q_min_bits}" if c.ConvLayer._q_schedule._q_min_bits != c.ConvLayer._out_bits else "")
                + (f" d{c.ConvLayer._dsp_count}" if c.ConvLayer._dsp_count > 0 else "")
                + ("+p" if c.PoolLayer is not None else "")
                + ")"
            )
            for c in cfg.layers
        )

        lcs.append(int(lc))
        fpss.append(round(fps, 2))
        hover.append(f"<b>#{i}</b>  [{int(lc)} LC  {fps:.1f} fps]<br>{layer_str}")

    fig = go.Figure(go.Scatter(
        x=lcs, y=fpss,
        mode="markers",
        text=hover,
        hoverinfo="text",
        marker=dict(size=6, opacity=0.75, color=lcs, colorscale="Viridis",
                    colorbar=dict(title="LC")),
    ))
    fig.update_layout(
        title="Feasible Networks: LC Utilization vs FPS",
        xaxis=dict(title="LC Utilization", range=[0.75*LC_CAP, LC_CAP]),
        yaxis=dict(title="FPS"),
        hovermode="closest",
    )
    fig.write_html(out_html)
    print(f"Plot saved → {out_html}")


if __name__ == "__main__":
    configs = generate_networks()
    print(f"\n{len(configs)} feasible networks (deduplicated)")
    plot_networks(configs)
