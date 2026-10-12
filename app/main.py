"""Dash demo of both models: punctuation / casing restoration and dialogue notes.

Run locally (uses ./release from ``uv run python -m src.publish.package`` if present, else the
Hub; override with ``MODEL2_SOURCE`` / ``MODEL1_SOURCE``)::

    uv run --extra app python -m app.main

In Docker, gunicorn serves ``app.main:create_server()`` (see ``Dockerfile``).
"""

from __future__ import annotations

import os
import textwrap
from typing import Protocol

import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html

from app.examples import CLEANUP, DIALOGUES
from app.text import Dialogue, changed_words, parse_dialogue
from src.model1_scorer.infer import ScoredDialogue
from src.model1_scorer.notes import HEADINGS, grouped

BLUE, LIGHT_BLUE, SURFACE, INK_2, GRID = "#2a78d6", "#b7d3f6", "#fcfcfb", "#52514e", "#e1e0d9"
TAG_LABELS = {
    "none": "",
    "key_point": "key point",
    "decision": "decision",
    "action_item": "action item",
}


class Backend(Protocol):
    """What the app needs from the models (see :class:`app.models.Models`)."""

    def clean(self, text: str) -> tuple[str, float]: ...

    def score(self, speakers: list[int], utterances: list[str]) -> tuple[ScoredDialogue, float]: ...


def model_badges(models: Backend) -> list[html.Span]:
    """One badge per loaded model: name, parameters and where it came from."""
    badges = []
    for label, bundle in (("Model 2 · cleanup", "cleanup"), ("Model 1 · notes", "scorer")):
        b = getattr(models, bundle, None)
        if b is None:
            continue
        src = b.config.get("source", {})
        run = f"{src.get('run', '?')} seed {src.get('seed', '?')}"
        text = f"{label}: {b.params / 1e6:.1f}M params · {run}"
        badges.append(html.Span(text, className="badge", title=f"Loaded from {b.source}"))
    return badges


def cleanup_tab() -> dcc.Tab:
    return dcc.Tab(
        label="Restore punctuation & casing",
        value="cleanup",
        children=[
            html.P(
                "Paste lowercase text without punctuation (a voice note, a transcript). Model 2 "
                "adds periods, commas, question marks and capital letters.",
                className="lead",
            ),
            dcc.Dropdown(
                id="clean-example",
                options=list(CLEANUP),
                placeholder="Load an example…",
                clearable=True,
            ),
            dcc.Textarea(id="clean-input", value=next(iter(CLEANUP.values())), className="input"),
            html.Button("Restore", id="clean-btn", n_clicks=0, className="primary"),
            dcc.Loading(html.Div(id="clean-output", className="output")),
        ],
    )


def notes_tab() -> dcc.Tab:
    return dcc.Tab(
        label="Dialogue notes",
        value="notes",
        children=[
            html.P(
                "One utterance per line, written as `Name: text`. Model 1 scores how important "
                "each utterance is, tags decisions and action items, and keeps the top three as "
                "notes.",
                className="lead",
            ),
            dcc.Dropdown(
                id="notes-example",
                options=list(DIALOGUES),
                placeholder="Load an example…",
                clearable=True,
            ),
            dcc.Textarea(
                id="notes-input", value=next(iter(DIALOGUES.values())), className="input tall"
            ),
            dcc.Checklist(
                id="notes-clean",
                options=[
                    {
                        "label": " Restore punctuation & casing first "
                        "(Model 2), for raw transcripts",
                        "value": "clean",
                    }
                ],
                value=[],
            ),
            html.Button("Make notes", id="notes-btn", n_clicks=0, className="primary"),
            dcc.Loading(html.Div(id="notes-output", className="output")),
        ],
    )


def render_cleanup(text: str, cleaned: str, ms: float) -> list:
    words = []
    for word, changed in changed_words(text, cleaned):
        words += [html.Mark(word) if changed else word, " "]
    return [
        html.H4("Result"),
        html.P(words, className="cleaned"),
        html.P(
            f"Highlighted: words the model capitalised or punctuated. {ms:.0f} ms on CPU.",
            className="meta",
        ),
    ]


def score_figure(dialogue: Dialogue, scored: ScoredDialogue) -> go.Figure:
    """One horizontal bar per utterance (dialogue order, top to bottom); picks in dark blue."""
    n = len(dialogue.utterances)
    labels = [f"{i + 1}. {dialogue.name(i)}" for i in range(n)]
    picked = set(scored.selected)
    tags = scored.tags or ["none"] * n
    hover = ["<br>".join(textwrap.wrap(u, 70)) for u in dialogue.utterances]
    fig = go.Figure(
        go.Bar(
            x=scored.scores,
            y=labels,
            orientation="h",
            marker_color=[BLUE if i in picked else LIGHT_BLUE for i in range(n)],
            text=[TAG_LABELS[t] for t in tags],
            textposition="outside",
            cliponaxis=False,
            textfont={"color": INK_2},
            customdata=hover,
            hovertemplate="%{y}: importance %{x:.2f}<br><br>%{customdata}<extra></extra>",
        )
    )
    fig.update_layout(
        height=34 * n + 90,
        margin={"l": 10, "r": 90, "t": 10, "b": 40},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        bargap=0.3,
        font={"family": "system-ui, -apple-system, 'Segoe UI', sans-serif", "color": INK_2},
        xaxis={
            "range": [0, 1],
            "title": "Importance (dark = kept in the notes)",
            "gridcolor": GRID,
        },
        yaxis={"autorange": "reversed", "automargin": True},
    )
    return fig


def render_notes(dialogue: Dialogue, scored: ScoredDialogue, ms: float, cleaned: bool) -> list:
    sections = []
    for tag, group in grouped(scored.selected, scored.tags):
        if scored.tags is not None:
            sections.append(html.H5(HEADINGS[tag]))
        sections.append(
            html.Ul(
                [
                    html.Li([html.Strong(f"{dialogue.name(i)}: "), dialogue.utterances[i]])
                    for i in group
                ]
            )
        )
    meta = f"{len(dialogue.utterances)} utterances, {ms:.0f} ms on CPU"
    meta += " (after punctuation restoration)." if cleaned else "."
    if scored.dropped:
        meta += f" {scored.dropped} utterance(s) did not fit in the model's 512 tokens."
    return [
        html.H4("Notes"),
        html.Div(sections, className="notes"),
        html.P(meta, className="meta"),
        html.H4("Importance of every utterance"),
        dcc.Graph(figure=score_figure(dialogue, scored), config={"displaylogo": False}),
    ]


def create_app(models: Backend) -> Dash:
    """The Dash app around ``models``."""
    app = Dash(__name__, title="Dialogue notes demo")
    app.layout = html.Main(
        [
            html.H1("Dialogue notes demo"),
            html.P(
                "Two small transformers trained from scratch with Keras 3 + JAX, running on CPU.",
                className="lead",
            ),
            html.Div(model_badges(models), className="badges"),
            dcc.Tabs(id="tabs", value="cleanup", children=[cleanup_tab(), notes_tab()]),
        ]
    )

    @app.callback(
        Output("clean-input", "value"), Input("clean-example", "value"), prevent_initial_call=True
    )
    def load_cleanup_example(name: str | None) -> str:
        return CLEANUP.get(name or "", "")

    @app.callback(
        Output("notes-input", "value"), Input("notes-example", "value"), prevent_initial_call=True
    )
    def load_dialogue_example(name: str | None) -> str:
        return DIALOGUES.get(name or "", "")

    @app.callback(
        Output("clean-output", "children"),
        Input("clean-btn", "n_clicks"),
        State("clean-input", "value"),
        prevent_initial_call=True,
    )
    def run_cleanup(_: int, text: str | None) -> list:
        if not text or not text.strip():
            return [html.P("Write some text first.", className="error")]
        cleaned, ms = models.clean(text)
        return render_cleanup(text, cleaned, ms)

    @app.callback(
        Output("notes-output", "children"),
        Input("notes-btn", "n_clicks"),
        State("notes-input", "value"),
        State("notes-clean", "value"),
        prevent_initial_call=True,
    )
    def run_notes(_: int, text: str | None, options: list[str] | None) -> list:
        try:
            dialogue = parse_dialogue(text or "")
        except ValueError as err:
            return [html.P(str(err), className="error")]
        clean_first = "clean" in (options or [])
        ms = 0.0
        if clean_first:
            cleaned = [models.clean(u) for u in dialogue.utterances]
            dialogue.utterances = [
                c or u for (c, _), u in zip(cleaned, dialogue.utterances, strict=True)
            ]
            ms += sum(t for _, t in cleaned)
        scored, score_ms = models.score(dialogue.speakers, dialogue.utterances)
        return render_notes(dialogue, scored, ms + score_ms, clean_first)

    return app


def create_server():
    """WSGI entry point for gunicorn: load the models from the environment, return Flask."""
    from app.models import Models

    return create_app(Models.from_env()).server


if __name__ == "__main__":
    from app.models import Models

    create_app(Models.from_env()).run(
        host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8050"))
    )
