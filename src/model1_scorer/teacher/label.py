"""Stage 6a: label DialogSum utterances with a Qwen teacher (PLAN.md section 5.3).

Usage (Kaggle GPU, in the vLLM environment; never with JAX/Keras)::

    python -m src.model1_scorer.teacher.label --config configs/teacher.yaml --teacher qwen14b

Per split, writes ``<output_dir>/<teacher>/<split>.jsonl`` (one validated dialogue per line:
raw scores on the teacher scale, scores normalised to [0, 1], tags, number of valid samples)
and ``failed_<split>.jsonl`` (dialogues still invalid after the retry, with the reason), then
``stats.json`` with failure rates. Lines are appended as dialogues finish and dialogues already
in either file are skipped, so a cut-off session resumes where it stopped.

Self-consistency: each dialogue is sampled ``generation.num_samples`` times at
``generation.temperature``; the valid answers are merged (mean score, majority tag, see
:func:`~src.model1_scorer.teacher.validate.combine_samples`). Only when no sample is valid, the
first answer is retried once with a correction message, decoded greedily. With
``use_reference: true`` the prompt includes the dialogue's reference summaries.

Backend: vLLM. If vLLM cannot be imported, the run falls back to ``transformers`` +
``bitsandbytes`` 4-bit on the non-AWQ checkpoint and logs a warning; the backend used is
recorded in ``stats.json``.

With ``train_subset: N`` only a fixed random subset of N train dialogues is labelled (the same
for every teacher: it depends only on ``seed``); validation and test are always labelled in full.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.config import load_config
from src.data.io import read_jsonl
from src.model1_scorer.teacher.prompt import build_messages, retry_messages
from src.model1_scorer.teacher.validate import (
    InvalidAnswer,
    TeacherLabels,
    combine_samples,
    validate_answer,
)

log = logging.getLogger(__name__)

# (conversations, sample) -> per conversation its answers (``num_samples`` sampled answers if
# ``sample`` else one greedy answer), or ``None`` when the prompt cannot be run (too long)
Generate = Callable[[list[list[dict[str, str]]], bool], list[list[str] | None]]
TOO_LONG = "prompt too long for the model context"


def vllm_backend(model: str, cfg: dict[str, Any]) -> Generate:
    """Batched chat generation with vLLM (``n`` samples share the prompt's prefill)."""
    from vllm import LLM, SamplingParams

    v = cfg["vllm"]
    llm = LLM(
        model=model,
        dtype=v["dtype"],
        max_model_len=v["max_model_len"],
        gpu_memory_utilization=v["gpu_memory_utilization"],
        tensor_parallel_size=v["tensor_parallel_size"],
        enable_prefix_caching=v["enable_prefix_caching"],
        seed=cfg["seed"],
    )
    tokenizer = llm.get_tokenizer()
    gen = cfg["generation"]
    max_new = gen["max_new_tokens"]
    sampled = SamplingParams(
        n=gen["num_samples"],
        temperature=gen["temperature"],
        top_p=gen["top_p"],
        max_tokens=max_new,
        seed=cfg["seed"],
    )
    greedy = SamplingParams(temperature=0.0, max_tokens=max_new)

    def generate(conversations: list[list[dict[str, str]]], sample: bool) -> list[list[str] | None]:
        lengths = [
            len(tokenizer.apply_chat_template(c, add_generation_prompt=True, tokenize=True))
            for c in conversations
        ]
        runnable = [i for i, n in enumerate(lengths) if n + max_new <= v["max_model_len"]]
        answers: list[list[str] | None] = [None] * len(conversations)
        if runnable:
            params = sampled if sample else greedy
            outputs = llm.chat([conversations[i] for i in runnable], params, use_tqdm=False)
            for i, out in zip(runnable, outputs, strict=True):
                answers[i] = [o.text for o in out.outputs]
        return answers

    return generate


def transformers_backend(model: str, cfg: dict[str, Any]) -> Generate:
    """Fallback: generation with ``transformers`` and a 4-bit ``bitsandbytes`` model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16")
    lm = AutoModelForCausalLM.from_pretrained(model, quantization_config=quant, device_map="auto")
    batch, gen = cfg["fallback"]["batch_size"], cfg["generation"]

    def generate(conversations: list[list[dict[str, str]]], sample: bool) -> list[list[str] | None]:
        n = gen["num_samples"] if sample else 1
        options = (
            {"do_sample": True, "temperature": gen["temperature"], "top_p": gen["top_p"]}
            if sample
            else {"do_sample": False}
        )
        answers: list[list[str] | None] = []
        for i in range(0, len(conversations), batch):
            texts = [
                tokenizer.apply_chat_template(c, add_generation_prompt=True, tokenize=False)
                for c in conversations[i : i + batch]
            ]
            enc = tokenizer(texts, return_tensors="pt", padding=True).to(lm.device)
            out = lm.generate(  # generate() runs without gradients
                **enc, **options, num_return_sequences=n, max_new_tokens=gen["max_new_tokens"]
            )
            texts = tokenizer.batch_decode(
                out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True
            )
            answers += [texts[j : j + n] for j in range(0, len(texts), n)]
        return answers

    return generate


def load_backend(teacher: str, cfg: dict[str, Any]) -> tuple[Generate, dict[str, str]]:
    """vLLM if it imports, else the transformers + bitsandbytes fallback."""
    try:
        import vllm  # noqa: F401
    except ImportError as err:
        model = cfg["fallback"]["models"][teacher]
        log.warning("vLLM unavailable (%s): FALLBACK to transformers 4-bit, model %s", err, model)
        return transformers_backend(model, cfg), {"backend": "transformers-bnb4", "model": model}
    model = cfg["teachers"][teacher]
    return vllm_backend(model, cfg), {"backend": "vllm", "model": model}


def select_dialogues(
    records: list[dict[str, Any]], split: str, subset: int | None, seed: int
) -> list[dict[str, Any]]:
    """All dialogues, or a fixed seeded subset of ``subset`` train dialogues (in file order)."""
    if split != "train" or subset is None or subset >= len(records):
        return records
    ids = sorted(r["id"] for r in records)
    keep = set(np.random.default_rng(seed).permutation(ids)[:subset].tolist())
    return [r for r in records if r["id"] in keep]


def done_ids(*paths: Path) -> set[str]:
    """IDs already written to any of ``paths`` (resume)."""
    return {r["id"] for p in paths if p.exists() for r in read_jsonl(p)}


def _append(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _check(answer: str | None, n: int, cfg: dict[str, Any]) -> TeacherLabels:
    if answer is None:
        raise InvalidAnswer(TOO_LONG)
    return validate_answer(answer, n, cfg["score_range"], cfg["tags"])


def _check_samples(answers: list[str] | None, n: int, cfg: dict[str, Any]) -> TeacherLabels:
    """Merge the valid answers; if none is valid, raise the first answer's problem."""
    if answers is None:
        raise InvalidAnswer(TOO_LONG)
    valid, first_error = [], None
    for a in answers:
        try:
            valid.append(_check(a, n, cfg))
        except InvalidAnswer as err:
            first_error = first_error or err
    if not valid:
        raise first_error or InvalidAnswer("no answer")
    return combine_samples(valid, cfg["score_range"])


def build_dialogue_messages(record: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, str]]:
    """Prompt for one DialogSum record (with its reference summaries if ``use_reference``)."""
    summaries = record["summaries"] if cfg["use_reference"] else None
    return build_messages(
        record["speakers"], record["utterances"], cfg["score_range"], cfg["tags"], summaries
    )


def label_dialogues(
    generate: Generate,
    records: Sequence[dict[str, Any]],
    labels_path: Path,
    failed_path: Path,
    cfg: dict[str, Any],
) -> dict[str, int]:
    """Label ``records`` not yet done, in chunks; one greedy retry with a correction message."""
    done = done_ids(labels_path, failed_path)
    todo = [r for r in records if r["id"] not in done]
    counts = {
        "skipped_done": len(records) - len(todo),
        "labelled": 0,
        "retried": 0,
        "failed": 0,
        "too_long": 0,
        "samples_generated": 0,
        "samples_valid": 0,
    }
    chunk = cfg["batch_dialogues"]
    for start in range(0, len(todo), chunk):
        batch = todo[start : start + chunk]
        convs = [build_dialogue_messages(r, cfg) for r in batch]
        answers = generate(convs, True)
        results: list[Any] = [None] * len(batch)
        errors: dict[int, str] = {}
        for i, (r, a) in enumerate(zip(batch, answers, strict=True)):
            counts["samples_generated"] += len(a or [])
            try:
                results[i] = _check_samples(a, len(r["utterances"]), cfg)
                counts["samples_valid"] += results[i].samples
            except InvalidAnswer as err:
                errors[i] = str(err)
        attempts = {i: list(answers[i] or [None]) for i in errors}
        retry = [i for i in errors if errors[i] != TOO_LONG] if cfg["max_retries"] else []
        if retry:
            counts["retried"] += len(retry)
            convs2 = [
                retry_messages(
                    convs[i],
                    answers[i][0],
                    errors[i],
                    len(batch[i]["utterances"]),
                    cfg["score_range"],
                    cfg["tags"],
                )
                for i in retry
            ]
            for i, a in zip(retry, generate(convs2, False), strict=True):
                try:
                    results[i] = _check_samples(a, len(batch[i]["utterances"]), cfg)
                    del errors[i]
                except InvalidAnswer as err:
                    errors[i] = f"{errors[i]}; retry: {err}"
                    attempts[i] += a or [None]
        good, bad = [], []
        for i, r in enumerate(batch):
            if i in errors:
                counts["too_long" if errors[i] == TOO_LONG else "failed"] += 1
                bad.append({"id": r["id"], "error": errors[i], "answers": attempts[i]})
            else:
                lab = results[i]
                good.append(
                    {
                        "id": r["id"],
                        "n": len(r["utterances"]),
                        "raw_scores": lab.raw_scores,
                        "scores": lab.scores,
                        "tags": lab.tags,
                        "samples": lab.samples,
                        "retried": i in retry,
                    }
                )
        _append(labels_path, good)
        _append(failed_path, bad)
        counts["labelled"] += len(good)
        log.info("%s: %d/%d done", labels_path.stem, start + len(batch), len(todo))
    return counts


def run(
    teacher: str,
    cfg: dict[str, Any],
    generate: Generate | None = None,
    backend_info: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Label every configured split with one teacher; returns (and writes) ``stats.json``."""
    if teacher not in cfg["teachers"]:
        raise ValueError(f"Unknown teacher {teacher!r}; known: {sorted(cfg['teachers'])}")
    if generate is None:
        generate, backend_info = load_backend(teacher, cfg)
    raw = Path(cfg["paths"]["raw_dir"]) / "dialogsum"
    out = Path(cfg["paths"]["output_dir"]) / teacher
    stats_path = out / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {"splits": {}}
    stats.update(teacher=teacher, **(backend_info or {}), config=cfg)
    for split in cfg["splits"]:
        start = time.time()
        records = select_dialogues(
            list(read_jsonl(raw / f"{split}.jsonl")), split, cfg["train_subset"], cfg["seed"]
        )
        counts = label_dialogues(
            generate, records, out / f"{split}.jsonl", out / f"failed_{split}.jsonl", cfg
        )
        n_labelled = len(done_ids(out / f"{split}.jsonl"))
        n_failed = len(done_ids(out / f"failed_{split}.jsonl"))
        stats["splits"][split] = {
            "dialogues": len(records),
            "labelled": n_labelled,
            "failed": n_failed,
            "failure_rate": round(n_failed / max(n_labelled + n_failed, 1), 4),
            "this_session": counts,
            "seconds_this_session": round(time.time() - start, 1),
        }
        log.info("%s/%s: %s", teacher, split, stats["splits"][split])
    stats["updated"] = datetime.now(UTC).isoformat(timespec="seconds")
    out.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/teacher.yaml"))
    parser.add_argument("--teacher", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(args.teacher, load_config(args.config))


if __name__ == "__main__":
    main()
