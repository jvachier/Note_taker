"""Tests for Model 2 word labels, sentence splitting and window packing."""

import pytest

from src.layers.losses import IGNORE
from src.model2_cleanup.labels import (
    CAPITALISED,
    CASING_CLASSES,
    COMMA,
    LOWER,
    NONE,
    PERIOD,
    PUNCT_CLASSES,
    QUESTION,
    UPPER,
    Word,
    casing_label,
    input_form,
    label_text,
    label_word,
    pack_windows,
    split_sentences,
)


def test_class_lists_match_plan():
    assert CASING_CLASSES == ("lower", "capitalised", "upper")
    assert PUNCT_CLASSES == ("none", "comma", "period", "question")


@pytest.mark.parametrize(
    ("core", "label"),
    [("hello", LOWER), ("Hello", CAPITALISED), ("I", CAPITALISED), ("a", LOWER),
     ("NASA", UPPER), ("iPhone", IGNORE), ("McDonald", IGNORE), ("2011", IGNORE),
     ("Well-known", CAPITALISED), ("Café", CAPITALISED), ("东京", IGNORE)],
)  # fmt: skip
def test_casing_label(core, label):
    assert casing_label(core) == label


@pytest.mark.parametrize(
    ("token", "word"),
    [
        ("Hello,", Word("hello", CAPITALISED, COMMA)),
        ("day.", Word("day", LOWER, PERIOD)),
        ("why?", Word("why", LOWER, QUESTION)),
        ("angry!", Word("angry", LOWER, PERIOD)),
        ("2011;", Word("2011", IGNORE, PERIOD)),
        ("paintings:", Word("paintings", LOWER, COMMA)),
        ('"Quoted,"', Word("quoted", CAPITALISED, COMMA)),
        ("(France)", Word("france", CAPITALISED, NONE)),
        ("end?).", Word("end", LOWER, QUESTION)),
        ("don't", Word("don't", LOWER, NONE)),
        ("1,000", Word("1,000", IGNORE, NONE)),
        ("3.5.", Word("3.5", IGNORE, PERIOD)),
        ("well-known", Word("well-known", LOWER, NONE)),
        ("Mr.", Word("mr", CAPITALISED, NONE)),
        ("U.S.", Word("us", UPPER, NONE)),
        ("U.S.,", Word("us", UPPER, COMMA)),
        ("J.", Word("j", CAPITALISED, NONE)),
        ("e.g.", Word("eg", LOWER, NONE)),
    ],
)
def test_label_word(token, word):
    assert label_word(token) == word


@pytest.mark.parametrize("token", ["-", "–", "&", "...", '"'])
def test_label_word_drops_tokens_without_letters_or_digits(token):
    assert label_word(token) is None


def test_input_form_is_lowercase_without_punctuation():
    assert input_form("Hello-World's") == "hello-world's"
    assert input_form("-start") == "start"
    assert input_form("a_b") == "ab"


def test_label_text_drops_lone_punctuation():
    words = label_text("In 2000 – 2005 , he left .")
    assert [w.form for w in words] == ["in", "2000", "2005", "he", "left"]


def _words(text):
    return label_text(text)


def test_split_sentences_at_period_and_question():
    sents = split_sentences(_words("Hi there. Mr. Smith is here? Yes, he is. trailing words"))
    assert [[w.form for w in s] for s in sents] == [
        ["hi", "there"],
        ["mr", "smith", "is", "here"],
        ["yes", "he", "is"],
        ["trailing", "words"],
    ]


def test_pack_windows_never_splits_a_sentence():
    sents = split_sentences(_words("A b c. D e. F g h i. Jo. K l m n o p q."))
    windows = pack_windows(sents, window_words=5)
    assert [len(w) for w in windows] == [5, 5, 7]
    flat = [w for window in windows for w in window]
    assert flat == [w for s in sents for w in s]
    for window in windows:
        assert window[-1].punct in (PERIOD, QUESTION)
