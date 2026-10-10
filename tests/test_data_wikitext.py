"""Tests for WikiText article handling and detokenisation."""

import pytest

from src.data.wikitext import detokenize, drop_articles, is_heading, iter_clean_text


def test_drop_articles_removes_whole_article():
    lines = ["", " = A = ", "", " a1 ", " = = Sec = = ", " a2 "]
    lines += [" = B = ", " b1 ", " = C = ", " c1 "]
    kept, n = drop_articles(lines, {"B"})
    assert n == 1
    assert kept == ["", " = A = ", "", " a1 ", " = = Sec = = ", " a2 ", " = C = ", " c1 "]


def test_drop_articles_counts_repeated_titles():
    kept, n = drop_articles([" = A = ", " x ", " = A = ", " y ", " = B = "], {"A"})
    assert (kept, n) == ([" = B = "], 2)


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        (" a well @-@ known , 1 @,@ 000 and 3 @.@ 5 . \n", "a well-known, 1,000 and 3.5."),
        (' he said " hello there " to them . ', 'he said "hello there" to them.'),
        (' " Quoted " start ', '"Quoted" start'),
        (" the game 's story ( 2011 ) ; it was ok ! ", "the game's story (2011); it was ok!"),
        (" I don 't know , I 'm sure we 'll see . ", "I don't know, I'm sure we'll see."),
        (" about 50 % of $ 5 million ? ", "about 50% of $5 million?"),
        ("", ""),
    ],
)
def test_detokenize(raw, clean):
    assert detokenize(raw) == clean


def test_is_heading():
    assert is_heading(" = Title = ")
    assert is_heading(" = = = Sub = = = \n")
    assert not is_heading(" a = b = c ")
    assert not is_heading("")


def test_iter_clean_text_skips_headings_and_blanks():
    lines = ["", " = A = ", " x , y . ", "  ", " = = S = = ", " z "]
    assert list(iter_clean_text(lines)) == ["x, y.", "z"]
