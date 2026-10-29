"""Tests for DialogSum parsing and test-reference grouping."""

import pytest

from src.data.dialogsum import (
    Dialogue,
    build_dialogue,
    count_empty_turns,
    group_test_rows,
    parse_dialogue,
)


def test_parse_strips_tags_and_keeps_speaker_ids():
    speakers, utts = parse_dialogue("#Person1#: Hi, Mr. Smith.\n#Person2#: Hello!\n#Person1#: Ok.")
    assert speakers == [1, 2, 1]
    assert utts == ["Hi, Mr. Smith.", "Hello!", "Ok."]


def test_parse_is_case_insensitive_and_skips_blank_lines():
    speakers, utts = parse_dialogue("#Person1#: a\n\n#PErson2#:   b  \r\n")
    assert speakers == [1, 2]
    assert utts == ["a", "b"]


def test_parse_joins_continuation_lines():
    speakers, utts = parse_dialogue("#Person1#: first part\nsecond part\n#Person3#: c")
    assert speakers == [1, 3]
    assert utts == ["first part second part", "c"]


def test_parse_keeps_mid_text_tags():
    _, utts = parse_dialogue("#Person1#: Ask #Person2# about it.")
    assert utts == ["Ask #Person2# about it."]


@pytest.mark.parametrize(
    "text",
    ["no tag here\n#Person1#: a", "#Person9#: too many speakers", "#Person1#:   ", "", "\n\n"],
    ids=["text-before-tag", "speaker-out-of-range", "only-empty-turn", "empty", "blank"],
)
def test_parse_rejects_malformed(text):
    with pytest.raises(ValueError):
        parse_dialogue(text, max_speakers=8)


def test_parse_drops_empty_turns():
    text = "#Person1#: a\n#Person3#: \n#Person2#: b\n#Person1#:\n#Person3#: c"
    speakers, utts = parse_dialogue(text)
    assert speakers == [1, 2, 3]
    assert utts == ["a", "b", "c"]
    assert count_empty_turns(text) == 2


def test_continuation_after_empty_tag_keeps_its_speaker():
    speakers, utts = parse_dialogue("#Person1#: a\n#Person2#:\nlate text")
    assert speakers == [1, 2]
    assert utts == ["a", "late text"]


def test_build_dialogue_and_round_trip():
    row = {"id": "train_0", "dialogue": "#Person1#: a\n#Person2#: b", "summary": " s "}
    row["topic"] = "t"
    d = build_dialogue(row, "train")
    assert d == Dialogue("train_0", "train", [1, 2], ["a", "b"], ["s"], ["t"])
    assert Dialogue.from_dict(d.to_dict()) == d


def _test_row(n, k, dialogue="#Person1#: a\n#Person2#: b"):
    return {"id": f"test_{n}_{k}", "dialogue": dialogue, "summary": f"s{n}{k}", "topic": f"t{k}"}


def test_group_test_rows_merges_references_in_order():
    rows = [
        _test_row(10, 2),
        _test_row(2, 1),
        _test_row(10, 1),
        _test_row(10, 3),
        _test_row(2, 3),
        _test_row(2, 2),
    ]
    out = group_test_rows(rows)
    assert [d.id for d in out] == ["test_2", "test_10"]
    assert out[1].summaries == ["s101", "s102", "s103"]
    assert out[1].topics == ["t1", "t2", "t3"]
    assert out[0].speakers == [1, 2] and out[0].split == "test"


def test_group_test_rows_rejects_mismatched_dialogues():
    rows = [_test_row(0, 1), _test_row(0, 2, dialogue="#Person1#: different")]
    with pytest.raises(ValueError, match="different dialogue"):
        group_test_rows(rows)


def test_group_test_rows_rejects_bad_ids():
    with pytest.raises(ValueError, match="Unexpected test ID"):
        group_test_rows([{"id": "test_0", "dialogue": "#Person1#: a", "summary": "", "topic": ""}])
