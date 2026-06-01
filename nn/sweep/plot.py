import re
from pathlib import Path

import plotly.graph_objects as go
import plotly.colors

from nn.config  import NNConfig
from nn.globals import LC_CAP, CLK_FREQ_HZ, NN_CFG
from nn.sweep.generate import generate_networks

RESULTS_PATH  = Path("profiling") / "nn_acc_pred" / "profiles" / "results.txt"

IMG_W = NN_CFG.input_dimensions.w
IMG_H = NN_CFG.input_dimensions.h
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
    fig.write_html(out_html)
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
        y=[r["hw_acc"] for r in aborted],
        mode="markers",
        name="aborted",
        text=[
            f"<b>{r['lc']} LC  {r['fps']:.1f} fps</b>  [aborted]<br>"
            f"hw_acc=0 (random chance)<br>{r['arch']}"
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
        y=[r["hw_acc"] for r in trained],
        mode="markers",
        name="trained",
        text=[
            f"<b>{r['lc']} LC  {r['fps']:.1f} fps</b><br>"
            f"hw_acc={r['hw_acc']:.4f}  float_acc={r['float_acc']:.4f}<br>{r['arch']}"
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
        title="Trained Networks: LUT4 Utilization vs Hardware Accuracy",
        xaxis=dict(title="LUT4 Utilization"),
        yaxis=dict(title="Hardware Accuracy", tickformat=".0%"),
        hovermode="closest",
    )
    fig.write_html(out_html)
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


if __name__ == "__main__":
    from nn.tasks.generate.generate import generate_networks
    configs = generate_networks()
    plot_networks(configs)
    plot_accuracy()
    plot_accuracy_by_bits()
