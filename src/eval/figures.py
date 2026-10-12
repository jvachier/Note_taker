"""Interactive Plotly figures for both models, built from :mod:`results_data`.

Each function returns one ``go.Figure`` without a title: the results report
(:mod:`src.eval.report`) puts the headline, the explanation and the verdict around it.
Colours follow one fixed system: grey = reference rows (lead-3, oracle, rule baseline),
orange = Qwen teachers, blue = our models.
"""

from __future__ import annotations

import textwrap
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.eval.results_data import (
    DATASET_NAMES,
    MODEL2_CLASSES,
    Model1Results,
    display_name,
    epoch_stats,
    group,
    model2_test_sets,
    model2_value,
    pooled_spearman,
    selection_positions,
)
from src.model1_scorer.evaluate import selection

# Reference palette (dataviz skill, light mode): categorical slots in fixed order, neutrals.
SURFACE, INK, INK_2, MUTED, GRID, AXIS = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SERIES = (BLUE, ORANGE, AQUA, YELLOW)
SEQUENTIAL = ("#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b")
GROUP_COLORS = {"reference": MUTED, "teacher": ORANGE, "student": BLUE}
GROUP_LABELS = {"reference": "Reference", "teacher": "Teacher (Qwen)", "student": "Student (ours)"}
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"


def style(fig: go.Figure, height: int) -> go.Figure:
    """Quiet shared layout: hairline grid, ink-token text, legend above the plot."""
    fig.update_layout(
        height=height,
        margin={"l": 10, "r": 30, "t": 40, "b": 10},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"family": FONT, "size": 13, "color": INK_2},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.06, "x": 0, "title": None},
        hoverlabel={"bgcolor": "white", "font": {"family": FONT, "color": INK}},
        bargap=0.35,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, automargin=True)
    fig.update_annotations(font={"size": 14, "color": INK})  # subplot titles
    return fig


def band_traces(
    stats: np.ndarray, color: str, name: str, show_legend: bool = True, fmt: str = ".3f"
) -> list[go.Scatter]:
    """Mean line with a ± std wash (one legend entry)."""
    x = np.arange(1, len(stats) + 1)
    lo, hi = stats[:, 0] - stats[:, 1], stats[:, 0] + stats[:, 1]
    rgb = tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))
    return [
        go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([hi, lo[::-1]]),
            fill="toself",
            mode="lines",
            fillcolor=f"rgba{(*rgb, 0.12)}",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
            legendgroup=name,
        ),
        go.Scatter(
            x=x,
            y=stats[:, 0],
            mode="lines",
            line={"color": color, "width": 2},
            name=name,
            legendgroup=name,
            showlegend=show_legend,
            customdata=stats[:, 1],
            hovertemplate=f"{name}<br>epoch %{{x}}: %{{y:{fmt}}} ± %{{customdata:{fmt}}}"
            "<extra></extra>",
        ),
    ]


# ----------------------------------------------------------------------------- Model 2


def model2_noise(seeds: dict[int, dict[str, Any]]) -> go.Figure:
    """Punctuation macro-F1 and casing accuracy against input noise, model vs rule baseline."""
    sets = model2_test_sets(seeds)
    metrics = [
        ("Punctuation macro-F1", ("punct", "macro_f1")),
        ("Casing accuracy", ("casing", "accuracy")),
    ]
    titles = [f"{DATASET_NAMES.get(d, d)}: {m}" for m, _ in metrics for d in sets]
    fig = make_subplots(
        rows=2, cols=len(sets), subplot_titles=titles, vertical_spacing=0.16, shared_xaxes=True
    )
    for col, (_dataset, entries) in enumerate(sets.items(), start=1):
        noise = [100 * n for n, _ in entries]
        for row, (label, path) in enumerate(metrics, start=1):
            first = row == 1 and col == 1
            base = [model2_value(seeds, k, "baseline", path)[0] for _, k in entries]
            model = np.array([model2_value(seeds, k, "model", path) for _, k in entries])
            fig.add_trace(
                go.Scatter(
                    x=noise,
                    y=base,
                    name="Rule baseline",
                    legendgroup="base",
                    showlegend=first,
                    mode="lines+markers",
                    line={"color": MUTED, "width": 2},
                    marker={"size": 8},
                    hovertemplate=f"Rule baseline<br>noise %{{x}}%<br>{label}: %{{y:.3f}}"
                    "<extra></extra>",
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=noise,
                    y=model[:, 0],
                    name="Model 2 (ours)",
                    legendgroup="model",
                    showlegend=first,
                    mode="lines+markers+text",
                    text=[""] * (len(noise) - 1) + [f"{model[-1, 0]:.2f}"],
                    textposition="middle right",
                    cliponaxis=False,
                    textfont={"color": INK_2},
                    error_y={"array": model[:, 1], "color": INK_2, "thickness": 1},
                    line={"color": BLUE, "width": 2},
                    marker={"size": 8},
                    customdata=model[:, 1],
                    hovertemplate=f"Model 2<br>noise %{{x}}%<br>{label}: %{{y:.3f}} ± "
                    "%{customdata:.3f}<extra></extra>",
                ),
                row=row,
                col=col,
            )
            fig.update_yaxes(range=[0, 1.05], row=row, col=col)
            fig.update_xaxes(tickvals=noise, ticksuffix="%", row=row, col=col)
    for col in range(1, len(sets) + 1):
        fig.update_xaxes(title_text="Word noise in the input", row=2, col=col)
    return style(fig, 620)


def model2_per_class(seeds: dict[int, dict[str, Any]], noise: float) -> go.Figure:
    """Per-class F1 (punctuation and casing) at one noise level, model vs rule baseline."""
    sets = model2_test_sets(seeds)
    fig = make_subplots(
        rows=1,
        cols=len(sets),
        shared_yaxes=True,
        subplot_titles=[DATASET_NAMES.get(d, d) for d in sets],
    )
    classes = [c for c, _ in MODEL2_CLASSES]
    for col, entries in enumerate(sets.values(), start=1):
        key = next(k for n, k in entries if np.isclose(n, noise))
        base = [model2_value(seeds, key, "baseline", p)[0] for _, p in MODEL2_CLASSES]
        model = np.array([model2_value(seeds, key, "model", p) for _, p in MODEL2_CLASSES])
        fig.add_trace(
            go.Bar(
                x=classes,
                y=base,
                name="Rule baseline",
                marker_color=MUTED,
                legendgroup="base",
                showlegend=col == 1,
                hovertemplate="Rule baseline<br>%{x}: F1 %{y:.3f}<extra></extra>",
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Bar(
                x=classes,
                y=model[:, 0],
                name="Model 2 (ours)",
                marker_color=BLUE,
                legendgroup="model",
                showlegend=col == 1,
                error_y={"array": model[:, 1], "color": INK_2, "thickness": 1},
                text=[f"{v:.2f}" for v in model[:, 0]],
                textposition="outside",
                textfont={"color": INK_2},
                customdata=model[:, 1],
                hovertemplate="Model 2<br>%{x}: F1 %{y:.3f} ± %{customdata:.3f}<extra></extra>",
            ),
            row=1,
            col=col,
        )
        fig.add_vline(x=2.5, line={"color": AXIS, "width": 1}, row=1, col=col)
        for x, label in ((1, "punctuation"), (3.5, "casing")):
            fig.add_annotation(
                x=x,
                y=1.13,
                text=label,
                showarrow=False,
                font={"color": MUTED, "size": 12},
                row=1,
                col=col,
            )
    fig.update_yaxes(range=[0, 1.18], title_text="F1", row=1, col=1)
    fig.update_layout(barmode="group", bargroupgap=0.08)
    return style(fig, 430)


def model2_training(histories: list[dict[str, np.ndarray]]) -> go.Figure:
    """Loss and validation macro-F1 per epoch (mean ± std over seeds)."""
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["Loss", "Validation punctuation macro-F1", "Validation casing macro-F1"],
        horizontal_spacing=0.07,
    )
    for trace in band_traces(epoch_stats(histories, "loss"), BLUE, "Train"):
        fig.add_trace(trace, row=1, col=1)
    for trace in band_traces(epoch_stats(histories, "val_loss"), ORANGE, "Validation"):
        fig.add_trace(trace, row=1, col=1)
    for col, key in ((2, "val_punct_macro_f1"), (3, "val_casing_macro_f1")):
        for trace in band_traces(epoch_stats(histories, key), BLUE, "Validation", False):
            fig.add_trace(trace, row=1, col=col)
    fig.update_xaxes(title_text="Epoch")
    return style(fig, 380)


# ----------------------------------------------------------------------------- Model 1


def _system_bar_traces(
    res: Model1Results,
    names: Sequence[str],
    value: Callable[[dict[str, Any]], float],
    label: str,
    show_legend: bool,
) -> list[go.Bar]:
    """One horizontal bar trace per system group (so the legend reads by group)."""
    traces = []
    for g in ("reference", "teacher", "student"):
        members = [n for n in names if group(n, res.students) == g]
        if not members:
            continue
        stats = np.array([res.value(n, value) for n in members])
        traces.append(
            go.Bar(
                y=[display_name(n) for n in members],
                x=stats[:, 0],
                orientation="h",
                name=GROUP_LABELS[g],
                legendgroup=g,
                showlegend=show_legend,
                marker_color=GROUP_COLORS[g],
                error_x={"array": stats[:, 1], "color": INK_2, "thickness": 1},
                text=[f"  {v:.3f}" for v in stats[:, 0]],
                textposition="outside",
                textfont={"color": INK_2},
                cliponaxis=False,
                customdata=stats[:, 1],
                hovertemplate=f"%{{y}}<br>{label}: %{{x:.3f}} ± %{{customdata:.3f}}<extra></extra>",
            )
        )
    return traces


def _system_panels(
    res: Model1Results,
    names: Sequence[str],
    panels: Sequence[tuple[str, Callable[[dict[str, Any]], float]]],
    height: int,
) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=len(panels),
        shared_yaxes=True,
        subplot_titles=[label for label, _ in panels],
        horizontal_spacing=0.06,
    )
    for col, (label, value) in enumerate(panels, start=1):
        for trace in _system_bar_traces(res, names, value, label, col == 1):
            fig.add_trace(trace, row=1, col=col)
    order = [display_name(n) for n in names][::-1]  # first system on top
    fig.update_yaxes(categoryorder="array", categoryarray=order, gridcolor=SURFACE)
    return style(fig, height)


def model1_quality(res: Model1Results) -> go.Figure:
    """ROUGE-1/2/L (and BERTScore when available) of the top-k notes, per system."""
    panels = [
        ("ROUGE-1", lambda m: m["rouge1"]),
        ("ROUGE-2", lambda m: m["rouge2"]),
        ("ROUGE-L", lambda m: m["rougeL"]),
    ]
    if all("bertscore" in m for ms in res.metrics.values() for m in ms):
        panels.append(("BERTScore F1", lambda m: m["bertscore"]))
    fig = _system_panels(res, list(res.runs), panels, 470)
    return fig


def model1_agreement(res: Model1Results, retention_target: float) -> go.Figure:
    """Retention and agreement with the reference teacher, per system."""
    names = [n for n in res.runs if n != res.reference]
    panels = [
        ("Retention (ROUGE vs teacher)", lambda m: m["retention"]),
        ("Spearman of scores vs teacher", lambda m: m["spearman"]),
        ("Top-3 overlap (Jaccard) vs teacher", lambda m: m["jaccard"]),
    ]
    fig = _system_panels(res, names, panels, 450)
    fig.add_vline(
        x=retention_target,
        line={"color": INK_2, "width": 1, "dash": "dot"},
        row=1,
        col=1,
        annotation_text=f"target {retention_target:.2f}",
        annotation_position="bottom right",
        annotation_font_color=INK_2,
    )
    fig.add_vline(x=0, line={"color": AXIS, "width": 1}, row=1, col=2)
    return fig


def model1_spearman_distribution(res: Model1Results) -> go.Figure:
    """Distribution over dialogues of the Spearman correlation with the reference teacher."""
    names = [n for n in res.runs if group(n, res.students) != "reference" and n != res.reference]
    fig = go.Figure()
    for name in names:
        g = group(name, res.students)
        fig.add_trace(
            go.Box(
                x=pooled_spearman(res, name),
                name=display_name(name),
                orientation="h",
                marker_color=GROUP_COLORS[g],
                boxpoints=False,
                line={"width": 1.5},
                fillcolor=GROUP_COLORS[g],
                opacity=0.85,
                showlegend=False,
            )
        )
    fig.add_vline(x=0, line={"color": AXIS, "width": 1})
    fig.update_xaxes(range=[-1.05, 1.05], title_text="Spearman correlation per dialogue")
    fig.update_yaxes(categoryorder="array", categoryarray=[display_name(n) for n in names][::-1])
    return style(fig, 90 * len(names) + 120)


def model1_tags(res: Model1Results) -> go.Figure:
    """``decision`` / ``action_item`` F1 against the reference teacher's tags."""
    names = [n for n in res.runs if n != res.reference and "tags" in res.metrics[n][0]]
    fig = go.Figure()
    for tag, label, color in (
        ("decision", "Decision", BLUE),
        ("action_item", "Action item", ORANGE),
    ):
        stats = np.array([res.value(n, lambda m, t=tag: m["tags"][t]["f1"]) for n in names])
        fig.add_trace(
            go.Bar(
                y=[display_name(n) for n in names],
                x=stats[:, 0],
                orientation="h",
                name=label,
                marker_color=color,
                error_x={"array": stats[:, 1], "color": INK_2, "thickness": 1},
                text=[f"  {v:.2f}" for v in stats[:, 0]],
                textposition="outside",
                textfont={"color": INK_2},
                cliponaxis=False,
                customdata=stats[:, 1],
                hovertemplate=f"%{{y}}<br>{label} F1: %{{x:.3f}} ± %{{customdata:.3f}}"
                "<extra></extra>",
            )
        )
    fig.update_layout(barmode="group", bargroupgap=0.08)
    fig.update_xaxes(range=[0, 1], title_text="F1 against the teacher's tags")
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=[display_name(n) for n in names][::-1],
        gridcolor=SURFACE,
    )
    return style(fig, 110 * len(names) + 120)


def model1_positions(res: Model1Results, systems: Sequence[str], bins: int) -> go.Figure:
    """Where in the dialogue each system picks its utterances (small multiples)."""
    systems = [s for s in systems if s in res.runs]
    fig = make_subplots(
        rows=1,
        cols=len(systems),
        shared_yaxes=True,
        subplot_titles=[display_name(s) for s in systems],
        horizontal_spacing=0.03,
    )
    labels = [f"{100 * b // bins}–{100 * (b + 1) // bins}" for b in range(bins)]
    for col, name in enumerate(systems, start=1):
        share = selection_positions(res, name, bins)
        fig.add_trace(
            go.Bar(
                x=labels,
                y=share,
                marker_color=GROUP_COLORS[group(name, res.students)],
                showlegend=False,
                name=display_name(name),
                hovertemplate=f"{display_name(name)}<br>%{{x}} of the dialogue: "
                "%{y:.0%} of picks<extra></extra>",
            ),
            row=1,
            col=col,
        )
    fig.update_yaxes(tickformat=".0%", title_text="Share of picked utterances", row=1, col=1)
    fig.update_xaxes(tickangle=0, tickfont={"size": 10})
    fig = style(fig, 360)
    fig.update_layout(margin={"b": 50})
    fig.add_annotation(
        text="Position in the dialogue (% from the start)",
        x=0.5,
        y=-0.16,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"color": INK_2, "size": 13},
    )
    return fig


def model1_example(res: Model1Results, dialogue_id: str) -> go.Figure:
    """Per-utterance scores of every system on one dialogue; hover shows the utterance."""
    d = res.dialogues[dialogue_id]
    n = len(d["utterances"])
    names = [s for s in res.runs if s != "Lead-3"]
    grid = np.zeros((len(names), n))
    texts = [
        "<br>".join(textwrap.wrap(f"P{s}: {u}", 60)[:4])
        for s, u in zip(d["speakers"], d["utterances"], strict=True)
    ]
    picks_x, picks_y = [], []
    for row, name in enumerate(names):
        entry = res.runs[name][0][dialogue_id]  # seed 0 for students
        scores = np.asarray(entry["scores"][:n], dtype=float)
        grid[row, : len(scores)] = scores
        for j in selection(entry, res.top_k):
            picks_x.append(j + 1)
            picks_y.append(display_name(name))
    ylabels = [display_name(s) for s in names]
    colorscale = [[i / len(SEQUENTIAL), c] for i, c in enumerate((SURFACE, *SEQUENTIAL))]
    fig = go.Figure(
        go.Heatmap(
            z=grid,
            x=list(range(1, n + 1)),
            y=ylabels,
            zmin=0,
            zmax=1,
            colorscale=colorscale,
            xgap=2,
            ygap=2,
            customdata=[texts] * len(names),
            colorbar={"title": {"text": "Score", "side": "right"}, "thickness": 12},
            hovertemplate="%{y}, utterance %{x}: score %{z:.2f}<br><br>%{customdata}"
            "<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=picks_x,
            y=picks_y,
            mode="markers",
            name="Picked (top-3)",
            marker={"size": 11, "color": "white", "line": {"color": INK, "width": 2}},
            hoverinfo="skip",
        )
    )
    fig.update_yaxes(autorange="reversed", gridcolor=SURFACE)
    fig.update_xaxes(
        title_text="Utterance (in dialogue order; hover to read it)", dtick=1, gridcolor=SURFACE
    )
    return style(fig, 60 * len(names) + 160)


def model1_size(res: Model1Results) -> go.Figure:
    """ROUGE-L against parameter count (log scale); lead-3 and oracle as reference lines."""
    fig = go.Figure()
    for g in ("teacher", "student"):
        members = [n for n, p in res.params.items() if p and group(n, res.students) == g]
        stats = np.array([res.value(n, lambda m: m["rougeL"]) for n in members])
        fig.add_trace(
            go.Scatter(
                x=[res.params[n] for n in members],
                y=stats[:, 0],
                mode="markers",
                name=GROUP_LABELS[g],
                marker={
                    "size": 13,
                    "color": GROUP_COLORS[g],
                    "line": {"color": SURFACE, "width": 2},
                },
                error_y={"array": stats[:, 1], "color": INK_2, "thickness": 1},
                text=[display_name(n) for n in members],
                hovertemplate="%{text}<br>%{x:.3s} parameters<br>ROUGE-L %{y:.3f}<extra></extra>",
            )
        )
    for name in ("Lead-3", "Oracle"):
        if name in res.runs:
            v = res.value(name, lambda m: m["rougeL"])[0]
            fig.add_hline(
                y=v,
                line={"color": MUTED, "width": 1, "dash": "dot"},
                annotation_text=f"{name} {v:.3f}",
                annotation_position="top left",
                annotation_font_color=INK_2,
            )
    # Students share one size: one label per cluster of near-equal scores.
    clusters: list[list[str]] = []
    for n in sorted(res.students, key=lambda n: res.value(n, lambda m: m["rougeL"])[0]):
        v = res.value(n, lambda m: m["rougeL"])[0]
        if clusters and v - res.value(clusters[-1][-1], lambda m: m["rougeL"])[0] < 0.006:
            clusters[-1].append(n)
        else:
            clusters.append([n])
    labels = [(c, ", ".join(display_name(n).removeprefix("Student ") for n in c)) for c in clusters]
    labels = [(c, f"Student{'s' if len(c) > 1 else ''} {t}") for c, t in labels]
    labels += [([n], display_name(n)) for n in res.params if n.startswith("Teacher ")]
    for members, text in labels:
        fig.add_annotation(
            x=np.log10(res.params[members[0]]),
            y=res.value(members[0], lambda m: m["rougeL"])[0],
            text=text,
            showarrow=False,
            xanchor="left",
            xshift=12,
            font={"color": INK_2},
        )
    fig.update_xaxes(type="log", title_text="Parameters (log scale)", range=[6.8, 10.5])
    fig.update_yaxes(title_text="ROUGE-L F1")
    return style(fig, 420)


def mlm_curves(history: dict[str, np.ndarray]) -> go.Figure:
    """MLM pretraining: loss and masked-token accuracy per epoch."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Loss", "Masked-token accuracy"])
    epochs = np.arange(1, len(history["loss"]) + 1)
    for col, key in ((1, "loss"), (2, "acc")):
        for split, prefix, color in (("Train", "", BLUE), ("Validation", "val_", ORANGE)):
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history[prefix + key],
                    name=split,
                    legendgroup=split,
                    showlegend=col == 1,
                    mode="lines",
                    line={"color": color, "width": 2},
                    hovertemplate=f"{split}<br>epoch %{{x}}: %{{y:.3f}}<extra></extra>",
                ),
                row=1,
                col=col,
            )
    fig.update_xaxes(title_text="Epoch")
    return style(fig, 360)


def student_training(histories: dict[str, list[dict[str, np.ndarray]]]) -> go.Figure:
    """Validation Spearman per epoch for every student (mean ± std over seeds)."""
    fig = go.Figure()
    for (name, hist), color in zip(histories.items(), SERIES, strict=False):
        for trace in band_traces(epoch_stats(hist, "val_spearman"), color, display_name(name)):
            fig.add_trace(trace)
    fig.update_xaxes(title_text="Epoch", dtick=1)
    fig.update_yaxes(title_text="Validation Spearman vs own targets")
    return style(fig, 400)
