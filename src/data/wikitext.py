"""WikiText-103 helpers: article boundaries and detokenisation of the raw text.

WikiText "raw" text is still tokenised: punctuation is space-separated (``word , word .``) and
hyphens / number separators are escaped as `` @-@ ``, `` @,@ ``, `` @.@ ``.
:func:`detokenize` turns a line back into ordinary text. Raw files on disk stay untouched;
consumers (tokenizer corpus, Model 2 examples) detokenise on read.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Iterator

# Top-level headings look like " = Title = "; sections use " = = Section = = ".
_TITLE_RE = re.compile(r"^ = (?!=)(.*?) = $")
_HEADING_RE = re.compile(r"^ (= )+.*( =)+ $")
_AT_RE = re.compile(r" @([-,.])@ ")

# Tokens glued to the previous token / to the next token. WikiText splits "don't" as "don 't".
_ATTACH_LEFT = {",", ".", ";", ":", "!", "?", ")", "]", "}", "%"}
_CLITIC_RE = re.compile(r"^'(s|t|m|d|re|ve|ll)$", re.IGNORECASE)
_ATTACH_RIGHT = {"(", "[", "{", "$", "£", "€"}


def article_title(line: str) -> str | None:
    """Return the article title if ``line`` is a top-level heading, else ``None``."""
    match = _TITLE_RE.match(line.rstrip("\n"))
    return match.group(1) if match else None


def article_titles(lines: Iterable[str]) -> list[str]:
    """Return the titles of all top-level headings, in order."""
    return [t for t in (article_title(line) for line in lines) if t is not None]


def is_heading(line: str) -> bool:
    """True for any article or section heading line (`` = T = ``, `` = = S = = `` ...)."""
    return bool(_HEADING_RE.match(line.rstrip("\n")))


def drop_articles(lines: Iterable[str], titles: Collection[str]) -> tuple[list[str], int]:
    """Remove every article whose title is in ``titles``.

    An article runs from its title line up to (excluding) the next title line. Lines before
    the first title are kept. Returns the kept lines and the number of articles dropped.
    """
    kept: list[str] = []
    dropping = False
    n_dropped = 0
    for line in lines:
        title = article_title(line)
        if title is not None:
            dropping = title in titles
            n_dropped += dropping
        if not dropping:
            kept.append(line)
    return kept, n_dropped


def detokenize(line: str) -> str:
    """Undo WikiText tokenisation on one line (headings are not special-cased).

    >>> detokenize(' a well @-@ known " test " ( 1 @,@ 000 people ) , isn \\'t it ? ')
    'a well-known "test" (1,000 people), isn\\'t it?'
    """
    tokens = _AT_RE.sub(r"\1", f" {line.strip()} ").split()
    out: list[str] = []
    glue_next = False
    quote_open = False
    for tok in tokens:
        if tok == '"':
            # Alternate opening / closing quotes within a line.
            attach = quote_open or glue_next
            quote_open = not quote_open
            glue_after = quote_open
        else:
            attach = glue_next or tok in _ATTACH_LEFT or bool(_CLITIC_RE.match(tok))
            glue_after = tok in _ATTACH_RIGHT
        if out and attach:
            out[-1] += tok
        else:
            out.append(tok)
        glue_next = glue_after
    return " ".join(out)


def iter_clean_text(lines: Iterable[str]) -> Iterator[str]:
    """Yield detokenised paragraph lines, skipping blank lines and headings."""
    for line in lines:
        if line.strip() and not is_heading(line):
            yield detokenize(line)
