"""Model 1 v3 corpus loading and pre-tokenisation (PLAN.md 10.2, 10.4)."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.config import deep_merge, load_config
from src.layers.losses import IGNORE
from src.model1_notes.data import (
    SPAN_TAGS,
    SPLITS,
    ZONES,
    NotesSet,
    bucket_for,
    build_split,
    encode_sentence,
    encode_split,
    gather_windows,
    read_corpus_split,
    sentence_rows,
    span_bio,
    speaker_numbers,
    window_starts,
)
from src.model1_notes.tiny_corpus import tiny_corpus_rows, write_tiny_corpus
from src.tokenizer.tokenizer import CLS_ID, PAD_ID, SEP_ID, load_tokenizer

DATA_CFG = {"max_sentence_tokens": 16, "max_sentences": 8, "window_overlap": 2}


@pytest.fixture(scope="module")
def tokenizer(tiny_assets):
    return load_tokenizer(tiny_assets["tokenizer"])


@pytest.fixture
def corpus(tmp_path):
    write_tiny_corpus(tmp_path / "corpus")
    return tmp_path / "corpus"


def test_read_groups_sorts_and_filters(tmp_path):
    layers = {
        "synthetic": tiny_corpus_rows("m1", "train")
        + tiny_corpus_rows("h1", "train", "synthetic_hn"),
        "ami": tiny_corpus_rows("m2", "train", "ami"),
    }
    for layer, rows in layers.items():
        out = tmp_path / "data" / layer / "sentences"
        out.mkdir(parents=True)
        rows = rows[::-1]  # file order must not matter within a meeting
        pq.write_table(pa.Table.from_pylist(rows), out / "train.parquet")
    meetings = read_corpus_split(tmp_path, "train", "sentence_text", ["synthetic", "ami"])
    assert list(meetings) == ["m1", "m2"]  # synthetic layer first, hard negatives filtered
    assert [r["sentence_id"] for r in meetings["m1"]] == list(range(6))
    assert meetings["m1"][0]["text"].startswith("Thanks")
    only_ami = read_corpus_split(tmp_path, "train", "sentence_text_asr", ["ami"])
    assert list(only_ami) == ["m2"] and "," not in only_ami["m2"][0]["text"]
    only_hn = read_corpus_split(tmp_path, "train", "sentence_text", ["synthetic_hn"])
    assert list(only_hn) == ["h1"]


def test_read_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_corpus_split(tmp_path, "train", "sentence_text", ["ami"])
    write_tiny_corpus(tmp_path)
    with pytest.raises(ValueError, match="Unknown sources"):
        read_corpus_split(tmp_path, "train", "sentence_text", ["dialogsum"])


def test_speaker_numbers_first_appearance_and_cap():
    assert speaker_numbers(["b", "a", "b", "c"]) == ([1, 2, 1, 3], 0)
    names = [f"p{i}" for i in range(10)]
    numbers, capped = speaker_numbers(names)
    assert numbers == [1, 2, 3, 4, 5, 6, 7, 8, 8, 8] and capped == 2


def test_span_bio():
    offsets = [(0, 3), (4, 8), (9, 12), (13, 15), (15, 18)]
    spans = [{"start": 0, "end": 3, "role": "owner"}, {"start": 13, "end": 18, "role": "deadline"}]
    tags = [SPAN_TAGS[t] for t in span_bio(offsets, spans)]
    assert tags == ["B-owner", "O", "O", "B-deadline", "I-deadline"]
    assert span_bio(offsets, None) == [0] * 5


def test_encode_sentence_layout_and_spans(tokenizer):
    text = "Ben will fix the test rig by Friday."
    spans = [{"start": 0, "end": 3, "role": "owner"}, {"start": 29, "end": 35, "role": "deadline"}]
    ids, tags, truncated = encode_sentence(tokenizer, text, spans, 32)
    n = int((ids != PAD_ID).sum())
    assert ids[0] == CLS_ID and ids[n - 1] == SEP_ID and not truncated
    assert tags[0] == IGNORE and tags[n - 1] == IGNORE and (tags[n:] == IGNORE).all()
    named = [SPAN_TAGS[t] for t in tags[1 : n - 1]]
    assert named[0] == "B-owner" and "B-deadline" in named
    assert named.count("B-owner") == 1 and named.count("B-deadline") == 1
    short_ids, short_tags, cut = encode_sentence(tokenizer, text, spans, 5)
    assert cut and short_ids[-1] == SEP_ID and short_tags[1] == SPAN_TAGS.index("B-owner")


@pytest.mark.parametrize(
    ("length", "expected"), [(3, [0]), (8, [0]), (9, [0, 1]), (20, [0, 6, 12])]
)
def test_window_starts(length, expected):
    starts = window_starts(length, 8, 2)
    assert starts == expected
    covered = set().union(*(range(s, min(s + 8, length)) for s in starts))
    assert covered == set(range(length))


def test_window_starts_rejects_bad_overlap():
    with pytest.raises(ValueError):
        window_starts(10, 8, 8)


def test_encode_split_arrays(tokenizer):
    meetings = {
        "a": read_rows("a", "synthetic", 1),
        "b": read_rows("b", "ami", 3),  # 18 sentences -> windows of 8 overlapping by 2
    }
    arrays, note_ids, counts = encode_split(tokenizer, meetings, DATA_CFG)
    assert note_ids == ["a", "b"]
    assert counts["sentences"] == 24 and counts["meetings"] == 2
    assert list(arrays["note_start"]) == [0, 6] and list(arrays["note_length"]) == [6, 18]
    assert list(arrays["source"]) == [0, 2]
    assert list(arrays["window_note"]) == [0, 1, 1, 1]
    assert list(arrays["window_offset"]) == [0, 0, 6, 10]
    assert list(arrays["window_length"]) == [6, 8, 8, 8]
    assert arrays["zone"][2] == ZONES.index("decision")
    assert list(arrays["speaker"][:6]) == [1, 2, 1, 2, 3, 1]
    assert list(arrays["sentence_id"][6:9]) == [0, 1, 2]
    assert arrays["confidence"].dtype == np.float32 and arrays["token_ids"].shape == (24, 16)


def read_rows(note_id, source, repeat):
    rows = tiny_corpus_rows(note_id, "train", source, repeat)
    for r in rows:
        r["text"] = r["sentence_text"]
    return rows


def test_build_split_load_and_gather(tmp_path, tokenizer, corpus):
    cfg = {
        "paths": {"corpus_dir": corpus},
        "corpus": {"text_field": "sentence_text", "sources": ["synthetic", "synthetic_hn", "ami"]},
        "data": {**DATA_CFG, "max_meetings": None},
    }
    counts = build_split(tokenizer, cfg, "train", tmp_path / "out")
    assert json.loads((tmp_path / "out" / "meta.json").read_text()) == counts
    ds = NotesSet.load(tmp_path / "out")
    assert ds.note_ids == ["train-0", "train-1", "train-2"]
    assert ds.num_windows == counts["windows"]
    batch = gather_windows(ds, [0, ds.num_windows - 1], slots=8)
    assert batch["token_ids"].shape == (2, 8, 16)
    assert batch["sentence_mask"].sum(axis=1).tolist() == [6, 8]
    pad = ~batch["sentence_mask"]
    assert (batch["importance"][pad] == IGNORE).all() and (batch["zone"][pad] == IGNORE).all()
    assert (batch["token_ids"][pad] == PAD_ID).all() and (batch["span_tags"][pad] == IGNORE).all()
    assert (batch["speaker"][pad] == 0).all() and (batch["rows"][pad] == -1).all()
    assert batch["source"].tolist() == [0, 2]
    assert batch["hard_negative"][0].sum() == 0
    with pytest.raises(ValueError):
        gather_windows(ds, [ds.num_windows - 1], slots=4)


def test_bucket_for():
    assert bucket_for(5, [32, 8, 16]) == 8 and bucket_for(16, [8, 16, 32]) == 16
    with pytest.raises(ValueError):
        bucket_for(33, [8, 16, 32])


def test_configs_load(configs_dir):
    for name in ("model1_v3", "model1_v3_E4", "model1_v3_E1", "model1_v3_tiny"):
        cfg = load_config(configs_dir / f"{name}.yaml")
        data = cfg["data"]
        assert max(data["sentence_buckets"]) == data["max_sentences"]
        assert 0 <= data["window_overlap"] < data["max_sentences"]
        assert data["max_sentence_tokens"] <= cfg["model"]["max_len"]
    e1 = load_config(configs_dir / "model1_v3_E1.yaml")
    assert e1["teachers"] == [] and e1["loss"]["qwen_kl"] == 0.0
    assert load_config(configs_dir / "model1_v3_E4.yaml")["teachers"] == ["qwen", "roberta"]


def test_tiny_pipeline_all_splits(tmp_path, tokenizer, corpus, configs_dir):
    cfg = deep_merge(
        load_config(configs_dir / "model1_v3_tiny.yaml"), {"paths": {"corpus_dir": corpus}}
    )
    for split in SPLITS:
        counts = build_split(tokenizer, cfg, split, tmp_path / split)
        assert counts["meetings"] == 3 and counts["sentences"] == 36


def test_sentence_rows_maps_keys_to_rows(tmp_path, tokenizer, corpus):
    cfg = {
        "paths": {"corpus_dir": corpus},
        "corpus": {"text_field": "sentence_text", "sources": ["synthetic", "synthetic_hn", "ami"]},
        "data": {**DATA_CFG, "max_meetings": None},
    }
    build_split(tokenizer, cfg, "test", tmp_path / "out")
    ds = NotesSet.load(tmp_path / "out")
    rows = sentence_rows(ds)
    assert len(rows) == len(ds["importance"])
    assert rows[("test-0", 0)] == 0 and rows[("test-1", 0)] == 6
