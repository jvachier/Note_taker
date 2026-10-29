"""Qwen3-0.6B labelling of the meeting corpus (PLAN.md 10.3). Runs in the teacher environment.

Usage (Notebook 1 runs the same code in its own venv; locally only ``--backend fake``)::

    python -m src.model1_notes.teacher.label --config run.json --backend fake --deadline <ts>

``run.json`` holds ``corpus_dir``, ``out_dir``, ``seed`` and ``teacher`` (the ``teacher``
section of the config: model, windows, budgets, which meetings to label).

**Generation, per meeting window** (vLLM):
1. *Thinking*: the chat prompt in Qwen3 thinking mode, sampled (``think_temperature``) until
   ``</think>`` or ``max_thinking_tokens`` (budget forcing: a cut-off thought is closed).
2. *Answer*: prompt + thought + ``</think>``, greedy, with the top ``top_logprobs``
   alternatives of every token, so :mod:`answer` can compute soft scores.
3. An invalid answer gets one greedy retry without thinking, with the reason. Still invalid:
   the window is recorded as failed and its sentences stay unlabelled.

**Long meetings** are cut into windows (:mod:`windows`); each sentence takes its label from the
window where it is farthest from the edges; ``depends_on`` becomes a ``sentence_id``.

**Outputs** in ``<out_dir>/qwen/`` (shard ``k`` of ``n`` writes ``shards/<k>/``):
    ``raw_<split>.jsonl``  one record per window (thought, answer, error) - analysis / blog
    ``<split>.jsonl``      one record per labelled meeting: per sentence ``sentence_id``,
                           ``score`` (soft), ``raw_score``, ``reason``, ``depends_on``
    ``stats.json``         counts, failure rate, seconds, ``complete``
:func:`merge_shards` writes the final ``<split>.parquet`` tables the student reads. Labelling
resumes: meetings already in ``<split>.jsonl`` or ``raw_<split>.jsonl`` are skipped.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.model1_notes.teacher.answer import (
    TOO_LONG,
    InvalidAnswer,
    Label,
    labels_from_answer,
    parse_entries,
)
from src.model1_notes.teacher.prompt import build_messages, retry_message
from src.model1_notes.teacher.windows import REASONS, window_starts

Tokens = list[tuple[str, dict[str, float]]]


@dataclass
class Generation:
    """One window's output: the thought (may be empty), the answer and its token log-probs."""

    thinking: str
    answer: str | None  # None: prompt too long for the context
    tokens: Tokens | None = None


#: ``generate(conversations, think) -> generations``; ``think=False`` answers directly.
Backend = Callable[[list[list[dict[str, str]]], bool], list[Generation]]


def read_meetings(corpus_dir: str | Path, split: str) -> dict[str, list[dict[str, Any]]]:
    """Sentences of every meeting of ``split`` (both layers), sorted by ``sentence_id``."""
    import pyarrow.parquet as pq

    columns = ["note_id", "source", "sentence_id", "speaker", "sentence_text"]
    meetings: dict[str, list[dict[str, Any]]] = {}
    for layer in ("synthetic", "ami"):
        path = Path(corpus_dir) / "data" / layer / "sentences" / f"{split}.parquet"
        if path.exists():
            for row in pq.read_table(path, columns=columns).to_pylist():
                meetings.setdefault(row["note_id"], []).append(row)
    for rows in meetings.values():
        rows.sort(key=lambda r: r["sentence_id"])
    return meetings


def select_meetings(
    meetings: dict[str, list[dict[str, Any]]], split: str, cfg: dict[str, Any], seed: int
) -> list[str]:
    """Meetings to label: all of them, except seeded subsets of the train meetings
    (``max_train_meetings`` synthetic, ``max_ami_train_meetings`` AMI; ``None`` = all)."""
    ids = list(meetings)
    if split != "train":
        return ids
    keep = set()
    for is_ami, key in ((False, "max_train_meetings"), (True, "max_ami_train_meetings")):
        group = [m for m in ids if (meetings[m][0]["source"] == "ami") == is_ami]
        cap = cfg.get(key)
        keep |= set(
            group if cap is None else random.Random(seed).sample(group, min(cap, len(group)))
        )
    return [m for m in ids if m in keep]


def in_shard(note_id: str, shard: int, num_shards: int) -> bool:
    """Stable assignment of a meeting to a shard (one shard per GPU)."""
    return zlib.crc32(note_id.encode()) % num_shards == shard


def label_windows(
    generate: Backend, windows: list[Sequence[dict[str, Any]]], min_coverage: float = 0.9
) -> tuple[list[dict[int, Label] | None], list[dict[str, Any]]]:
    """Labels per window (``id -> Label``, possibly partial; ``None`` = nothing usable) and one
    raw record per window. A window below ``min_coverage`` gets one greedy retry without
    thinking; the better of the two answers is kept."""
    convs = [
        build_messages([r["speaker"] for r in w], [r["sentence_text"] for r in w]) for w in windows
    ]
    first = generate(convs, True)
    labels: list[dict[int, Label] | None] = [None] * len(windows)
    raw: list[dict[str, Any]] = []
    retry, errors = [], {}
    for i, (w, g) in enumerate(zip(windows, first, strict=True)):
        try:
            labels[i] = labels_from_answer(g.answer, len(w), REASONS, g.tokens, min_coverage)
        except InvalidAnswer as err:
            errors[i] = str(err)
            if g.answer is not None and str(err) != TOO_LONG:
                retry.append(i)
            partial = parse_entries(g.answer, len(w), REASONS)
            labels[i] = labels_from_answer(g.answer, len(w), REASONS, g.tokens) if partial else None
        raw.append({"thinking": g.thinking, "answer": g.answer, "error": errors.get(i)})
    if retry:
        convs2 = [
            [
                *convs[i],
                {"role": "assistant", "content": first[i].answer or ""},
                retry_message(errors[i], len(windows[i]), REASONS),
            ]
            for i in retry
        ]
        for i, g in zip(retry, generate(convs2, False), strict=True):
            n = len(windows[i])
            again = (
                labels_from_answer(g.answer, n, REASONS, g.tokens)
                if parse_entries(g.answer, n, REASONS)
                else None
            )
            if again and len(again) > len(labels[i] or {}):
                labels[i] = again
            covered = len(labels[i] or {})
            error = None if covered >= min_coverage * n else f"{errors[i]}; retry: {covered}/{n}"
            raw[i].update(retried=True, retry_answer=g.answer, error=error)
    for r, lab, w in zip(raw, labels, windows, strict=True):
        r["labelled"] = len(lab or {})
        r["sentences"] = len(w)
    return labels, raw


def merge_meeting(
    rows: Sequence[dict[str, Any]],
    starts: Sequence[int],
    window_labels: Sequence[dict[int, Label] | None],
    size: int,
) -> list[dict[str, Any]] | None:
    """Per-sentence labels of a meeting from its (possibly partial) windows.

    Each sentence takes its label from the window that labelled it where it is farthest from
    the window edges; sentences no window labelled are left out. ``None`` if nothing is
    labelled.
    """
    sentences = []
    for i, row in enumerate(rows):
        best, best_margin = None, -1
        for w, (start, labels) in enumerate(zip(starts, window_labels, strict=True)):
            local = i - start + 1
            if labels and start <= i < start + size and local in labels:
                margin = min(i - start, start + size - 1 - i)
                if margin > best_margin:
                    best, best_margin = w, margin
        if best is None:
            continue
        start, label = starts[best], window_labels[best][i - starts[best] + 1]  # type: ignore[index]
        dep = -1 if label.depends_on < 0 else rows[start + label.depends_on - 1]["sentence_id"]
        sentences.append(
            {
                "sentence_id": row["sentence_id"],
                "score": label.soft,
                "raw_score": label.score,
                "reason": label.reason,
                "depends_on": dep,
            }
        )
    return sentences or None


def done_ids(*paths: Path) -> set[str]:
    """Meeting IDs already written to the given JSONL files."""
    ids = set()
    for p in paths:
        if p.exists():
            ids |= {json.loads(line)["note_id"] for line in p.open() if line.strip()}
    return ids


def label_split(
    generate: Backend,
    meetings: dict[str, list[dict[str, Any]]],
    ids: Sequence[str],
    out: Path,
    split: str,
    cfg: dict[str, Any],
    deadline: float,
) -> dict[str, Any]:
    """Label ``ids`` of one split into ``out`` (resumable). Returns this split's counts."""
    size, overlap = cfg["window_sentences"], cfg["window_overlap"]
    good_p, raw_p = out / f"{split}.jsonl", out / f"raw_{split}.jsonl"
    todo = [m for m in ids if m not in done_ids(good_p, raw_p)]
    start, n_good, n_failed = time.time(), 0, 0
    for b in range(0, len(todo), cfg["batch_meetings"]):
        if time.time() > deadline:
            return {"stopped_early": True, "labelled": n_good, "failed": n_failed}
        batch = todo[b : b + cfg["batch_meetings"]]
        starts = {m: window_starts(len(meetings[m]), size, overlap) for m in batch}
        jobs = [(m, s) for m in batch for s in starts[m]]
        windows = [meetings[m][s : s + size] for m, s in jobs]
        labels, raw = label_windows(generate, windows, cfg.get("min_window_coverage", 0.9))
        by_meeting: dict[str, list[dict[int, Label] | None]] = {m: [] for m in batch}
        with raw_p.open("a") as rf:
            for (m, s), lab, r in zip(jobs, labels, raw, strict=True):
                by_meeting[m].append(lab)
                rf.write(json.dumps({"note_id": m, "start": s, **r}) + "\n")
        with good_p.open("a") as gf:
            for m in batch:
                merged = merge_meeting(meetings[m], starts[m], by_meeting[m], size)
                if merged is None:
                    n_failed += 1
                    continue
                n_good += 1
                gf.write(json.dumps({"note_id": m, "sentences": merged}) + "\n")
        print(f"qwen/{split}: {b + len(batch)}/{len(todo)}", flush=True)
    return {
        "stopped_early": False,
        "labelled": n_good,
        "failed": n_failed,
        "seconds": round(time.time() - start, 1),
    }


def fake_backend(seed: int = 0) -> Backend:
    """Smoke runs and tests only: a keyword rule with made-up log-probabilities, never Qwen."""

    def answer_for(conv: list[dict[str, str]]) -> Generation:
        lines = conv[1]["content"].split("sentences):\n", 1)[1].splitlines()
        entries = []
        for i, line in enumerate(lines, 1):
            text = line.split(": ", 1)[1].lower()
            if " will " in f" {text} ":
                score, reason = 3, "commitment_with_owner"
            elif "hold" in text or "decid" in text:
                score, reason = 3, "agreed_outcome"
            elif "failed" in text:
                score, reason = 2, "risk_or_blocker"
            else:
                score, reason = 0, "filler"
            dep = i - 1 if reason == "agreed_outcome" and i > 1 else -1
            entries.append(f'[{i}, {score}, "{reason}", {dep}]')
        pieces = ["[", *(e + ", " for e in entries[:-1]), entries[-1], "]"]
        tokens: Tokens = []
        for piece in pieces:  # the score digit is its own token, with alternatives
            head, sep, rest = piece.partition(", ")
            tokens.append((head + sep, {}))
            if rest:
                digit, rest2 = rest[0], rest[1:]
                alt = {digit: -0.1, str((int(digit) + 1) % 4): -2.5}
                tokens += [(digit, alt), (rest2, {})]
        text = "".join(t for t, _ in tokens)
        return Generation("Thinking about who decided what.", text, tokens)

    def generate(convs: list[list[dict[str, str]]], think: bool) -> list[Generation]:
        return [answer_for(c[:2]) for c in convs]

    return generate


def vllm_backend(cfg: dict[str, Any], seed: int) -> Backend:  # pragma: no cover - Kaggle only
    """Qwen3 through vLLM: budget-forced thinking, then a greedy answer with log-probs."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=cfg["model"],
        dtype="half",
        max_model_len=cfg["max_model_len"],
        gpu_memory_utilization=cfg["gpu_memory_utilization"],
        enable_prefix_caching=True,
        seed=seed,
    )
    tok = llm.get_tokenizer()
    think = SamplingParams(
        temperature=cfg["think_temperature"],
        top_p=cfg["top_p"],
        max_tokens=cfg["max_thinking_tokens"],
        stop=["</think>"],
        seed=seed,
    )
    answer = SamplingParams(
        temperature=0.0, max_tokens=cfg["max_answer_tokens"], logprobs=cfg["top_logprobs"]
    )

    def tokens_of(out: Any) -> Tokens:
        tokens = []
        for tid, alts in zip(out.token_ids, out.logprobs or [], strict=False):
            chosen = alts[tid].decoded_token if tid in alts else ""
            tokens.append((chosen, {a.decoded_token: a.logprob for a in alts.values()}))
        return tokens

    def generate(convs: list[list[dict[str, str]]], use_thinking: bool) -> list[Generation]:
        prompts = [
            tok.apply_chat_template(
                c, tokenize=False, add_generation_prompt=True, enable_thinking=use_thinking
            )
            for c in convs
        ]
        budget = cfg["max_answer_tokens"] + (cfg["max_thinking_tokens"] if use_thinking else 0)
        fits = [len(tok.encode(p)) + budget <= cfg["max_model_len"] for p in prompts]
        results = [Generation("", None) for _ in convs]
        ok = [i for i, f in enumerate(fits) if f]
        thoughts = {i: "" for i in ok}
        if use_thinking and ok:
            for i, out in zip(ok, llm.generate([prompts[i] for i in ok], think), strict=True):
                thoughts[i] = out.outputs[0].text
        second = []
        for i in ok:
            if use_thinking:
                thought = thoughts[i]
                if not thought.lstrip().startswith("<think>"):
                    thought = "<think>\n" + thought
                second.append(prompts[i] + thought.rstrip() + "\n</think>\n\n")
            else:
                second.append(prompts[i])
        for i, out in zip(ok, llm.generate(second, answer), strict=True):
            tokens = tokens_of(out.outputs[0])
            text = "".join(t for t, _ in tokens) or out.outputs[0].text
            results[i] = Generation(thoughts[i], text, tokens)
        return results

    return generate


def write_stats(path: Path, stats: dict[str, Any]) -> None:
    path.write_text(json.dumps(stats, indent=2) + "\n")


def run(run_cfg: dict[str, Any], backend: str, deadline: float, shard: int, num_shards: int) -> int:
    """Label every configured split for one shard. Returns 0 (done or stopped at deadline)."""
    cfg, seed = run_cfg["teacher"], run_cfg["seed"]
    out = Path(run_cfg["out_dir"]) / "qwen" / "shards" / str(shard)
    out.mkdir(parents=True, exist_ok=True)
    if backend == "fake":
        generate, info = fake_backend(seed), {"backend": "fake (smoke only)", "model": None}
    else:  # pragma: no cover - Kaggle only
        try:
            import vllm

            generate = vllm_backend(cfg, seed)
            info = {"backend": f"vllm {vllm.__version__}", "model": cfg["model"]}
        except Exception as err:  # not installed, or cannot start on this GPU
            print(f"vLLM unusable: {type(err).__name__}: {err}", flush=True)
            return 3
    stats_path = out / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {"splits": {}}
    stats.update(info, complete=False)
    for split in cfg["splits"]:
        meetings = read_meetings(run_cfg["corpus_dir"], split)
        ids = [
            m for m in select_meetings(meetings, split, cfg, seed) if in_shard(m, shard, num_shards)
        ]
        result = label_split(generate, meetings, ids, out, split, cfg, deadline)
        prev = stats["splits"].get(split, {})
        result["seconds"] = round(prev.get("seconds", 0) + result.get("seconds", 0), 1)
        result["meetings"] = len(ids)
        result["labelled"] = len(done_ids(out / f"{split}.jsonl"))
        result["failed"] = len(done_ids(out / f"raw_{split}.jsonl")) - result["labelled"]
        result["failure_rate"] = round(result["failed"] / max(len(ids), 1), 4)
        stats["splits"][split] = result
        write_stats(stats_path, stats)
        if result["stopped_early"]:
            print("deadline reached, stopping (resume next session)", flush=True)
            return 0
    stats["complete"] = True
    write_stats(stats_path, stats)
    return 0


def merge_shards(out_dir: str | Path, num_shards: int, splits: Sequence[str]) -> dict[str, int]:
    """``qwen/shards/*/<split>.jsonl`` -> ``qwen/<split>.parquet`` (one row per sentence)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(out_dir) / "qwen"
    counts = {}
    for split in splits:
        rows = []
        for k in range(num_shards):
            path = root / "shards" / str(k) / f"{split}.jsonl"
            if path.exists():
                for line in path.open():
                    if line.strip():
                        rec = json.loads(line)
                        rows += [{"note_id": rec["note_id"], **s} for s in rec["sentences"]]
        rows.sort(key=lambda r: (r["note_id"], r["sentence_id"]))
        schema = pa.schema(
            [
                ("note_id", pa.string()),
                ("sentence_id", pa.int32()),
                ("score", pa.float32()),
                ("raw_score", pa.int8()),
                ("reason", pa.string()),
                ("depends_on", pa.int32()),
            ]
        )
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), root / f"{split}.parquet")
        counts[split] = len(rows)
    return counts


def main(argv: list[str] | None = None) -> None:
    """Entry point (see module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--backend", choices=["vllm", "fake"], default="vllm")
    parser.add_argument("--deadline", type=float, default=float("inf"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args(argv)
    run_cfg = json.loads(args.config.read_text())
    sys.exit(run(run_cfg, args.backend, args.deadline, args.shard, args.num_shards))


if __name__ == "__main__":
    main()
