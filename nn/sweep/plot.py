import re
from pathlib import Path

import plotly.graph_objects as go
import plotly.colors

from nn.config  import NNConfig
from nn.globals import LC_CAP, CLK_FREQ_HZ, NN_CFG
from nn.sweep.generate import generate_networks

RESULTS_PATH  = Path("profiling") / "nn_acc_pred" / "profiles" / "results.txt"

IMG_W = NN_CFG.in_dims.width
IMG_H = NN_CFG.in_dims.height
PLOT_HTML     = Path("nn") / "sweep" / "networks_plot.html"
ACCURACY_HTML = Path("nn") / "sweep" / "accuracy_plot.html"
BITS_HTML     = Path("nn") / "sweep" / "bits_plot.html"

_PAT_TRAINED = re.compile(
    r"^\s*\d+\s+([\d.]+)\s+([\d.]+)\s+\[(\d+)\s+LC\s+([\d.]+)\s+fps\]\s+(.+)$"
)
_PAT_ABORTED = re.compile(
    r"^\s*\d+\s+aborted\s+\[(\d+)\s+LC\s+([\d.]+)\s+fps\]\s+(.+)$"
)
_PAT_WB = re.compile(r"conv\(\s*\d+ch\s+(\d+)b")


def _wb_seq(arch: str) -> tuple[int, ...]:
    """Return the activation bit-width sequence across all conv layers."""
    return tuple(int(b) for b in _PAT_WB.findall(arch))


def _seq_label(seq: tuple[int, ...]) -> str:
    return "→".join(f"{b}b" for b in seq)


def _parse_results(results_path: str) -> list[dict]:
    """Parse results.txt into a list of records.

    Each record has: lc, fps, hw_acc, float_acc, arch, aborted, wb, wb_seq, seq_label.
    wb is the first-layer activation bit-width; wb_seq is the full non-decreasing
    sequence across all layers; seq_label is its human-readable string (e.g. "2b→3b").
    Aborted records have hw_acc=0 and float_acc=None.
    """
    records = []
    with open(results_path) as f:
        for line in f:
            m = _PAT_TRAINED.match(line)
            if m:
                arch = m.group(5).strip()
                seq  = _wb_seq(arch)
                records.append({
                    "float_acc": float(m.group(1)),
                    "hw_acc":    float(m.group(2)),
                    "lc":        int(m.group(3)),
                    "fps":       float(m.group(4)),
                    "arch":      arch,
                    "aborted":   False,
                    "wb":        seq[0] if seq else None,
                    "wb_seq":    seq,
                    "seq_label": _seq_label(seq),
                })
                continue
            m = _PAT_ABORTED.match(line)
            if m:
                arch = m.group(3).strip()
                seq  = _wb_seq(arch)
                records.append({
                    "float_acc": None,
                    "hw_acc":    0,
                    "lc":        int(m.group(1)),
                    "fps":       float(m.group(2)),
                    "arch":      arch,
                    "aborted":   True,
                    "wb":        seq[0] if seq else None,
                    "wb_seq":    seq,
                    "seq_label": _seq_label(seq),
                })
    return records


def plot_networks(
    configs: list[tuple[int | float, NNConfig]],
    out_html: str = str(PLOT_HTML),
) -> None:
    """Scatter plot of LC vs FPS for all enumerated networks."""
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
        title="Feasible Networks: LUT4 Utilization vs FPS",
        xaxis=dict(title="LUT4 Utilization", range=[0.75 * LC_CAP, LC_CAP]),
        yaxis=dict(title="FPS"),
        hovermode="closest",
    )
    fig.write_html(out_html, include_plotlyjs='cdn')
    print(f"Plot saved → {out_html}")


def plot_accuracy(
    results_path: str = str(RESULTS_PATH),
    out_html: str = str(ACCURACY_HTML),
) -> None:
    """Scatter plot of LC utilization vs hw accuracy, colored by FPS.
    Aborted runs are plotted at hw_acc=0 (random chance baseline)."""
    records = _parse_results(results_path)
    trained = [r for r in records if not r["aborted"]]
    aborted = [r for r in records if r["aborted"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[r["lc"] for r in aborted],
        y=[r["float_acc"] for r in aborted],
        mode="markers",
        name="aborted",
        text=[
            f"<b>{r['lc']} LC  {r['fps']:.1f} fps</b>  [aborted]<br>"
            f"float_acc=0.10 (random chance)<br>{r['arch']}"
            for r in aborted
        ],
        hoverinfo="text",
        marker=dict(
            size=6, symbol="x", opacity=0.45,
            color=[r["fps"] for r in aborted],
            colorscale="Plasma", showscale=False,
        ),
    ))
    fig.add_trace(go.Scatter(
        x=[r["lc"] for r in trained],
        y=[r["float_acc"] for r in trained],
        mode="markers",
        name="trained",
        text=[
            f"<b>{r['lc']} LC  {r['fps']:.1f} fps</b><br>"
            f"float_acc={r['float_acc']:.4f}  hw_acc={r['hw_acc']:.4f}<br>{r['arch']}"
            for r in trained
        ],
        hoverinfo="text",
        marker=dict(
            size=8, opacity=0.85,
            color=[r["fps"] for r in trained],
            colorscale="Plasma",
            colorbar=dict(title="FPS"),
            showscale=True,
        ),
    ))
    fig.update_layout(
        title="Trained Networks: LC Utilization vs Float Accuracy",
        xaxis=dict(title="LC Utilization"),
        yaxis=dict(title="Float Accuracy", tickformat=".0%"),
        hovermode="closest",
    )
    fig.write_html(out_html, include_plotlyjs='cdn')
    print(f"Accuracy plot saved → {out_html}  ({len(trained)} trained, {len(aborted)} aborted)")


def plot_accuracy_by_bits(
    results_path: str = str(RESULTS_PATH),
    out_html: str = str(BITS_HTML),
) -> None:
    """LC utilization vs hw accuracy, one trace per activation bit-width sequence.

    Each unique wb sequence (e.g. "2b→3b", "3b→3b→4b") becomes its own colored
    trace. Sequences are sorted by (min_wb, max_wb, length) so flat low-precision
    families appear first and growing sequences appear last.
    Trained networks are solid circles; aborted runs are faint x markers at 0.
    """
    records = _parse_results(results_path)

    # Sort sequences: flat low-precision first, growing last
    seqs = sorted(
        {r["wb_seq"] for r in records if r["wb_seq"]},
        key=lambda s: (min(s), max(s), len(s)),
    )
    colors = plotly.colors.qualitative.Plotly + plotly.colors.qualitative.Dark24

    fig = go.Figure()
    for i, seq in enumerate(seqs):
        label  = _seq_label(seq)
        color  = colors[i % len(colors)]
        group   = [r for r in records if r["wb_seq"] == seq]
        trained = [r for r in group if not r["aborted"]]
        aborted = [r for r in group if r["aborted"]]

        if aborted:
            fig.add_trace(go.Scatter(
                x=[r["lc"] for r in aborted],
                y=[r["hw_acc"] for r in aborted],
                mode="markers",
                name=f"{label} [aborted]",
                legendgroup=label,
                showlegend=False,
                text=[
                    f"<b>{r['lc']} LC  {r['fps']:.1f} fps</b>  [aborted]<br>"
                    f"{label}  •  random chance<br>{r['arch']}"
                    for r in aborted
                ],
                hoverinfo="text",
                marker=dict(size=6, symbol="x", opacity=0.3, color=color),
            ))

        if trained:
            fig.add_trace(go.Scatter(
                x=[r["lc"] for r in trained],
                y=[r["hw_acc"] for r in trained],
                mode="markers",
                name=label,
                legendgroup=label,
                text=[
                    f"<b>{r['lc']} LC  {r['fps']:.1f} fps</b><br>"
                    f"{label}<br>"
                    f"hw_acc={r['hw_acc']:.4f}  float_acc={r['float_acc']:.4f}<br>{r['arch']}"
                    for r in trained
                ],
                hoverinfo="text",
                marker=dict(size=8, opacity=0.85, color=color),
            ))

    fig.update_layout(
        title="Hardware Accuracy vs LUT4 Utilization by Activation Bit-Width Sequence",
        xaxis=dict(title="LUT4 Utilization"),
        yaxis=dict(title="Hardware Accuracy", tickformat=".0%"),
        hovermode="closest",
        legend=dict(title="wb sequence (click to isolate)"),
    )

    # Click a legend item to isolate that family; click again to restore.
    # Returns false to suppress Plotly's default show/hide toggle.
    _CLICK_JS = """
(function() {
    var gd = document.querySelector('.plotly-graph-div');
    var selected = null;
    var origOpacities = gd.data.map(function(t) {
        return (t.marker && t.marker.opacity != null) ? t.marker.opacity : 0.85;
    });

    gd.on('plotly_legendclick', function(data) {
        var group = data.data[data.curveNumber].legendgroup;
        if (selected === group) {
            Plotly.restyle(gd, 'marker.opacity', origOpacities);
            selected = null;
        } else {
            selected = group;
            Plotly.restyle(gd, 'marker.opacity', gd.data.map(function(t, i) {
                return t.legendgroup === group ? origOpacities[i] : 0.04;
            }));
        }
        return false;
    });

    gd.on('plotly_legenddoubleclick', function() {
        Plotly.restyle(gd, 'marker.opacity', origOpacities);
        selected = null;
        return false;
    });
})();
"""
    fig.write_html(out_html, post_script=_CLICK_JS)
    print(f"Bits plot saved → {out_html}  ({len(seqs)} sequences)")


def plot_accuracy_by_bandwidth(
    results_path: str = str(RESULTS_PATH),
    out_html: str = str(Path("nn") / "sweep" / "bandwidth_plot.html"),
) -> None:
    """Scatter plot of total (channels * in_bits) vs float accuracy."""
    records = _parse_results(results_path)
    
    first_in_bits = NN_CFG.layers[0].ConvLayer._in_bits if hasattr(NN_CFG, "layers") and len(NN_CFG.layers) > 0 else 1

    # Calculate total channels * in_bits for each record and assign 10% acc to aborted
    for r in records:
        arch_str = r["arch"]
        channels = [int(x) for x in re.findall(r'(\d+)ch', arch_str)]
        bits = [int(x) for x in re.findall(r'(\d+)b', arch_str)]
        
        in_bits_list = [first_in_bits] + bits[:-1]
        r["bandwidth"] = sum(c * b for c, b in zip(channels, in_bits_list))

        if r["aborted"]:
            r["plot_acc"] = 0.10
        else:
            r["plot_acc"] = r["float_acc"]

    trained = [r for r in records if not r["aborted"]]
    aborted = [r for r in records if r["aborted"]]

    fig = go.Figure()
    
    if aborted:
        fig.add_trace(go.Scatter(
            x=[r["bandwidth"] for r in aborted],
            y=[r["plot_acc"] for r in aborted],
            mode="markers",
            name="aborted",
            text=[
                f"<b>{r['bandwidth']} Bandwidth (Ch*Bits)</b><br>"
                f"acc=10% (random chance)<br>{r['arch']}"
                for r in aborted
            ],
            hoverinfo="text",
            marker=dict(size=6, symbol="x", opacity=0.45, color="red"),
        ))
        
    if trained:
        fig.add_trace(go.Scatter(
            x=[r["bandwidth"] for r in trained],
            y=[r["plot_acc"] for r in trained],
            mode="markers",
            name="trained",
            text=[
                f"<b>{r['bandwidth']} Bandwidth (Ch*Bits)</b><br>"
                f"float_acc={r['plot_acc']:.4f}<br>{r['arch']}"
                for r in trained
            ],
            hoverinfo="text",
            marker=dict(size=8, opacity=0.85, color="blue"),
        ))

    fig.update_layout(
        title="Accuracy vs Total Bandwidth (Log Scale)",
        xaxis=dict(title="Σ (Output Channels × Input Bits)", type="log"),
        yaxis=dict(title="Float Accuracy", tickformat=".0%"),
        hovermode="closest",
    )
    fig.write_html(out_html, include_plotlyjs='cdn')
    print(f"Bandwidth plot saved → {out_html}")

def plot_accuracy_by_ternary(
    results_path: str = str(RESULTS_PATH),
    out_html: str = str(Path("nn") / "sweep" / "ternary_plot.html"),
) -> None:
    """Scatter plot of % ternary datapath vs float accuracy."""
    records = _parse_results(results_path)
    
    for r in records:
        arch_str = r["arch"]
        bits = [int(x) for x in re.findall(r'(\d+)b', arch_str)]
        
        r["pct_ternary"] = sum(1 for b in bits if b == 2) / len(bits)

        if r["aborted"]:
            r["plot_acc"] = 0.10
        else:
            r["plot_acc"] = r["float_acc"]

    trained = [r for r in records if not r["aborted"]]
    aborted = [r for r in records if r["aborted"]]

    fig = go.Figure()
    
    if aborted:
        fig.add_trace(go.Scatter(
            x=[r["pct_ternary"] for r in aborted],
            y=[r["plot_acc"] for r in aborted],
            mode="markers",
            name="aborted",
            text=[
                f"<b>{r['pct_ternary']:.1%} Ternary Layers</b><br>"
                f"acc=10% (random chance)<br>{r['arch']}"
                for r in aborted
            ],
            hoverinfo="text",
            marker=dict(size=6, symbol="x", opacity=0.45, color="red"),
        ))
        
    if trained:
        fig.add_trace(go.Scatter(
            x=[r["pct_ternary"] for r in trained],
            y=[r["plot_acc"] for r in trained],
            mode="markers",
            name="trained",
            text=[
                f"<b>{r['pct_ternary']:.1%} Ternary Layers</b><br>"
                f"float_acc={r['plot_acc']:.4f}<br>{r['arch']}"
                for r in trained
            ],
            hoverinfo="text",
            marker=dict(size=8, opacity=0.85, color="blue"),
        ))

    fig.update_layout(
        title="Accuracy vs % Ternary Datapath",
        xaxis=dict(title="% of Feature Layers using 2-bit Output", tickformat=".0%"),
        yaxis=dict(title="Float Accuracy", tickformat=".0%"),
        hovermode="closest",
    )
    fig.write_html(out_html, include_plotlyjs='cdn')
    print(f"Ternary plot saved → {out_html}")

def plot_lc_vs_fps(
    results_path: str = str(RESULTS_PATH),
    out_html: str = str(Path("nn") / "sweep" / "lc_vs_fps_plot.html"),
) -> None:
    """Scatter plot of LC utilization vs FPS."""
    records = _parse_results(results_path)
    
    # Exclude aborted runs as they have 0 FPS, or maybe include them but colored differently.
    # Actually, the user wants to see LC vs FPS. Aborted runs don't have valid FPS, so we should filter them out.
    trained = [r for r in records if not r["aborted"]]

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=[r["lc"] for r in trained],
        y=[r["fps"] for r in trained],
        mode="markers",
        name="trained",
        text=[
            f"<b>{r['lc']} LC  {r['fps']:.1f} fps</b><br>"
            f"float_acc={r['float_acc']:.4f}<br>{r['arch']}"
            for r in trained
        ],
        hoverinfo="text",
        marker=dict(
            size=8, opacity=0.85,
            color=[r["float_acc"] for r in trained],
            colorscale="Viridis",
            colorbar=dict(title="Float Acc"),
            showscale=True,
        ),
    ))

    fig.update_layout(
        title="Logic Cell Utilization vs FPS",
        xaxis=dict(title="Logic Cell (LC) Utilization"),
        yaxis=dict(title="Frames Per Second (FPS)"),
        hovermode="closest",
    )
    fig.write_html(out_html, include_plotlyjs='cdn')
    print(f"LC vs FPS plot saved → {out_html}")

def plot_depth_vs_fps(
    results_path: str = str(RESULTS_PATH),
    out_html: str = str(Path("nn") / "sweep" / "depth_vs_fps_plot.html"),
) -> None:
    """Scatter plot of Network Depth vs FPS, with average accuracy trend."""
    import re
    from plotly.subplots import make_subplots
    records = _parse_results(results_path)
    
    trained = [r for r in records if not r["aborted"]]
    
    depth_accs = {}
    for r in trained:
        arch_str = r["arch"]
        channels = [int(x) for x in re.findall(r'(\d+)ch', arch_str)]
        r["depth"] = len(channels)
        depth_accs.setdefault(r["depth"], []).append(r["float_acc"])

    # Calculate average accuracy per depth
    depths = sorted(depth_accs.keys())
    avg_accs = [sum(depth_accs[d])/len(depth_accs[d]) for d in depths]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Scatter(
        x=[r["depth"] for r in trained],
        y=[r["fps"] for r in trained],
        mode="markers",
        name="Networks",
        text=[
            f"<b>Depth: {r['depth']} layers  {r['fps']:.1f} fps</b><br>"
            f"float_acc={r['float_acc']:.4f}<br>{r['arch']}"
            for r in trained
        ],
        hoverinfo="text",
        marker=dict(
            size=8, opacity=0.85,
            color=[r["float_acc"] for r in trained],
            colorscale="Viridis",
            colorbar=dict(title="Float Acc", x=1.1),
            showscale=True,
        ),
    ), secondary_y=False)

    # Add trend line for average accuracy
    fig.add_trace(go.Scatter(
        x=depths,
        y=avg_accs,
        mode="lines+markers",
        name="Avg Accuracy",
        line=dict(color="red", width=3, dash="dash"),
        marker=dict(size=10, symbol="diamond"),
        text=[f"Avg Acc: {acc:.4f} at Depth {d}" for d, acc in zip(depths, avg_accs)],
        hoverinfo="text"
    ), secondary_y=True)

    fig.update_layout(
        title="Network Depth vs FPS (with Average Accuracy Trend)",
        xaxis=dict(title="Network Depth (Number of Conv Layers)"),
        hovermode="closest",
    )
    
    fig.update_yaxes(title_text="Frames Per Second (FPS)", secondary_y=False)
    fig.update_yaxes(title_text="Average Float Accuracy", tickformat=".0%", secondary_y=True)
    
    fig.write_html(out_html, include_plotlyjs='cdn')
    print(f"Depth vs FPS plot saved → {out_html}")

if __name__ == "__main__":
    from nn.sweep.generate import generate_networks
    configs = generate_networks()
    plot_networks(configs)
    plot_accuracy()
    plot_accuracy_by_bits()
    plot_accuracy_by_bandwidth()
    plot_accuracy_by_ternary()
    plot_lc_vs_fps()
    plot_depth_vs_fps()
