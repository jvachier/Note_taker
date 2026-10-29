"""Tests for split-leakage checks and WikiText article detection."""

import pytest

from src.data.splits import assert_disjoint, find_overlaps
from src.data.wikitext import article_title, article_titles


def test_find_overlaps_reports_each_pair():
    out = find_overlaps({"train": ["a", "b"], "validation": ["c"], "test": ["b", "c"]})
    assert out == {("test", "train"): {"b"}, ("test", "validation"): {"c"}}


def test_assert_disjoint_passes_on_disjoint_splits():
    assert_disjoint({"train": ["train_0", "train_1"], "validation": ["dev_0"], "test": ["test_0"]})


def test_assert_disjoint_raises_on_shared_id():
    with pytest.raises(ValueError, match="more than one split"):
        assert_disjoint({"train": ["x", "y"], "test": ["y"]})


@pytest.mark.parametrize(
    ("line", "title"),
    [
        (" = Valkyria Chronicles III = \n", "Valkyria Chronicles III"),
        (" = The Hustler ( film ) = ", "The Hustler ( film )"),
        (" = = Gameplay = = \n", None),
        (" = = = Release = = = ", None),
        (" Some text with = signs = inside . ", None),
        ("", None),
    ],
)
def test_article_title(line, title):
    assert article_title(line) == title


def test_article_titles_in_order():
    lines = ["", " = A = ", " text ", " = = Sec = = ", " = B = "]
    assert article_titles(lines) == ["A", "B"]


def test_dialogue_text_overlap_ignores_speakers():
    from src.data.dialogsum import Dialogue
    from src.data.download import dialogue_text_overlap

    dialogues = {
        "train": [
            Dialogue("train_0", "train", [1, 2], ["a", "b"]),
            Dialogue("train_1", "train", [1], ["c"]),
        ],
        "validation": [Dialogue("dev_0", "validation", [2, 1], ["a", "b"])],
        "test": [Dialogue("test_0", "test", [1], ["d"])],
    }
    assert dialogue_text_overlap(dialogues) == {"train/validation": [[["train_0"], ["dev_0"]]]}


def test_drop_train_duplicates_only_touches_train():
    from src.data.dialogsum import Dialogue
    from src.data.download import drop_train_duplicates

    dialogues = {
        "train": [Dialogue(f"train_{i}", "train", [1], [t]) for i, t in enumerate("abc")],
        "validation": [Dialogue("dev_0", "validation", [1], ["a"])],
        "test": [Dialogue("test_0", "test", [1], ["c"]), Dialogue("test_1", "test", [1], ["a"])],
    }
    assert drop_train_duplicates(dialogues, ["validation", "test"]) == ["train_0", "train_2"]
    assert [d.id for d in dialogues["train"]] == ["train_1"]
    assert len(dialogues["validation"]) == 1 and len(dialogues["test"]) == 2
