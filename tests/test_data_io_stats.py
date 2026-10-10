"""Tests for raw-data IO helpers and descriptive statistics."""

from src.data.dialogsum import Dialogue
from src.data.io import read_jsonl, read_lines, write_jsonl, write_lines
from src.data.stats import dialogsum_stats, wikitext_stats


def test_jsonl_round_trip_keeps_unicode(tmp_path):
    records = [{"a": 1, "t": "戦場 café"}, {"a": 2, "t": ""}]
    assert write_jsonl(tmp_path / "x" / "f.jsonl", records) == 2
    assert list(read_jsonl(tmp_path / "x" / "f.jsonl")) == records
    assert "戦場" in (tmp_path / "x" / "f.jsonl").read_text(encoding="utf-8")


def test_lines_round_trip_keeps_blank_lines(tmp_path):
    lines = ["", " = A = \n", " text \n", ""]
    assert write_lines(tmp_path / "f.txt", lines) == 4
    assert list(read_lines(str(tmp_path / "f.txt"))) == ["", " = A = ", " text ", ""]


def test_dialogsum_stats():
    ds = [
        Dialogue("a", "train", [1, 2, 1], ["one two", "three", "four five six"], ["s"], ["t"]),
        Dialogue("b", "train", [1, 2], ["x", "y z"], ["s"], ["t"]),
    ]
    st = dialogsum_stats(ds)
    assert st["dialogues"] == 2
    assert st["utterances_per_dialogue"]["max"] == 3
    assert st["words_per_dialogue"]["mean"] == 4.5
    assert st["utterances_by_speaker"] == {"1": 3, "2": 2}


def test_wikitext_stats():
    st = wikitext_stats(["", " = A = ", " one two . ", " = = S = = ", " three "])
    assert st == {"lines": 5, "nonempty_lines": 4, "articles": 1, "words": 12}
