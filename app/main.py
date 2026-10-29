"""Dash demo of both models: punctuation / casing restoration and meeting notes (Model 1 v3).

Run locally (uses ./release from ``uv run python -m src.publish.package`` if present, else the
Hub; override with ``MODEL2_SOURCE`` / ``MODEL1_SOURCE``)::

    uv run --extra app python -m app.main

In Docker, gunicorn serves ``app.main:create_server()`` (see ``Dockerfile``).
"""

from __future__ import annotations

import logging
import os
import textwrap
from typing import Protocol

import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html

from app.examples import CLEANUP, DIALOGUES
from app.text import MAX_CHARS, changed_words, check_length, meeting_sentences, parse_dialogue
from src.model1_notes.notes import SECTION_OF_ZONE, SECTIONS, Note

BLUE, LIGHT_BLUE, SURFACE, INK_2, GRID = "#2a78d6", "#b7d3f6", "#fcfcfb", "#52514e", "#e1e0d9"
ZONE_LABELS = {"decision": "decision", "action_item": "action item"}
MODEL_ERROR = "Something went wrong while running the model. Please try again with another text."
log = logging.getLogger(__name__)


class Backend(Protocol):
    """What the app needs from the models (see :class:`app.models.Models`)."""

    def clean(self, text: str) -> tuple[str, float]: ...

    def annotate(self, speakers: list[str], sentences: list[str]) -> tuple[list[Note], float]: ...


def model_badges(models: Backend) -> list[html.Span]:
    """One badge per loaded model: name, parameters and where it came from."""
    badges = []
    for label, bundle in (("Model: cleanup", "cleanup"), ("Model: meeting notes", "notes")):
        b = getattr(models, bundle, None)
        if b is None:
            continue
        text = f"{label}: {b.params / 1e6:.1f}M params"
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
            dcc.Textarea(
                id="clean-input",
                value=next(iter(CLEANUP.values())),
                maxLength=MAX_CHARS,
                className="input",
            ),
            html.Button("Restore", id="clean-btn", n_clicks=0, className="primary"),
            dcc.Loading(html.Div(id="clean-output", className="output")),
        ],
    )


def notes_tab() -> dcc.Tab:
    return dcc.Tab(
        label="Meeting notes",
        value="notes",
        children=[
            html.P(
                "One turn per line, written as `Name: text`. Model 1 reads the whole meeting, "
                "scores every sentence, keeps the important ones and sorts them into decisions, "
                "action items (with owner and deadline) and key points.",
                className="lead",
            ),
            dcc.Dropdown(
                id="notes-example",
                options=list(DIALOGUES),
                placeholder="Load an example…",
                clearable=True,
            ),
            dcc.Textarea(
                id="notes-input",
                value=next(iter(DIALOGUES.values())),
                maxLength=MAX_CHARS,
                className="input tall",
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
                value=["clean"],
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


def score_figure(notes: list[Note], threshold: float) -> go.Figure:
    """One horizontal bar per sentence (meeting order, top to bottom); kept ones in dark blue."""
    labels = [f"{n.index + 1}. {n.speaker}" for n in notes]
    hover = ["<br>".join(textwrap.wrap(n.text, 70)) for n in notes]
    fig = go.Figure(
        go.Bar(
            x=[n.score for n in notes],
            y=labels,
            orientation="h",
            marker_color=[BLUE if n.kept else LIGHT_BLUE for n in notes],
            text=[ZONE_LABELS.get(n.zone, "") if n.kept else "" for n in notes],
            textposition="outside",
            cliponaxis=False,
            textfont={"color": INK_2},
            customdata=hover,
            hovertemplate="%{y}: importance %{x:.2f}<br><br>%{customdata}<extra></extra>",
        )
    )
    fig.add_vline(x=threshold, line_dash="dot", line_color=INK_2)
    fig.update_layout(
        height=28 * len(notes) + 90,
        margin={"l": 10, "r": 90, "t": 10, "b": 40},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        bargap=0.3,
        font={"family": "system-ui, -apple-system, 'Segoe UI', sans-serif", "color": INK_2},
        xaxis={
            "range": [0, 1],
            "title": f"Importance (dark = kept; dotted line = threshold {threshold:.2f})",
            "gridcolor": GRID,
        },
        yaxis={"autorange": "reversed", "automargin": True},
    )
    return fig


def note_item(note: Note) -> html.Li:
    """One bullet: speaker, sentence, for action items owner and deadline, and the replies
    attached to it (confirmations)."""
    parts = [html.Strong(f"{note.speaker}: "), note.text]
    if note.zone == "action_item" and (note.owner or note.deadline):
        parts.append(
            html.Span(
                f" (owner: {note.owner or '?'}, due: {note.deadline or '?'})", className="meta"
            )
        )
    if note.followups:
        parts.append(
            html.Ul(
                [
                    html.Li([html.Strong(f"↳ {r.speaker}: "), r.text], className="meta")
                    for r in note.followups
                ]
            )
        )
    return html.Li(parts)


def render_notes(notes: list[Note], threshold: float, ms: float, cleaned: bool) -> list:
    sections: dict[str, list[Note]] = {name: [] for name in SECTIONS}
    for note in notes:
        if note.kept:
            sections[SECTION_OF_ZONE.get(note.zone, "Key points")].append(note)
    blocks = []
    for name, items in sections.items():
        blocks.append(html.H5(name))
        blocks.append(
            html.Ul([note_item(n) for n in items])
            if items
            else html.P("None found.", className="meta")
        )
    kept = sum(n.kept for n in notes)
    meta = f"{kept} of {len(notes)} sentences kept, {ms:.0f} ms on CPU"
    meta += " (after punctuation restoration)." if cleaned else "."
    return [
        html.H4("Notes"),
        html.Div(blocks, className="notes"),
        html.P(meta, className="meta"),
        html.H4("Importance of every sentence"),
        dcc.Graph(figure=score_figure(notes, threshold), config={"displaylogo": False}),
    ]


def released_threshold(models: Backend) -> float:
    """The keep threshold tuned on validation and shipped in the model's ``config.json``."""
    notes = getattr(models, "notes", None)
    return float(notes.config["inference"]["threshold"]) if notes is not None else 0.5


def create_app(models: Backend) -> Dash:
    """The Dash app around ``models``."""
    app = Dash(__name__, title="Meeting notes demo")
    app.layout = html.Main(
        [
            html.H1("Meeting notes demo"),
            html.P(
                "Two small transformers trained from scratch with Keras 3 + JAX, running on CPU.",
                className="lead",
            ),
            html.Div(model_badges(models), className="badges"),
            dcc.Tabs(
                id="tabs",
                value="cleanup",
                children=[cleanup_tab(), notes_tab()],
            ),
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
        try:
            check_length(text)
        except ValueError as err:
            return [html.P(str(err), className="error")]
        try:
            cleaned, ms = models.clean(text)
        except Exception:
            log.exception("cleanup failed")
            return [html.P(MODEL_ERROR, className="error")]
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
        try:
            ms = 0.0
            if clean_first:
                cleaned = [models.clean(u) for u in dialogue.utterances]
                dialogue.utterances = [
                    c or u for (c, _), u in zip(cleaned, dialogue.utterances, strict=True)
                ]
                ms += sum(t for _, t in cleaned)
            speakers, sentences = meeting_sentences(dialogue)
            notes, notes_ms = models.annotate(speakers, sentences)
        except Exception:
            log.exception("notes failed")
            return [html.P(MODEL_ERROR, className="error")]
        return render_notes(notes, released_threshold(models), ms + notes_ms, clean_first)

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
