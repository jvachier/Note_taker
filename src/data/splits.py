"""Checks that official splits do not leak into each other."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import combinations


def find_overlaps(items_by_split: Mapping[str, Iterable[str]]) -> dict[tuple[str, str], set[str]]:
    """Return the shared items for every pair of splits that share at least one item."""
    sets = {name: set(items) for name, items in items_by_split.items()}
    overlaps = {}
    for a, b in combinations(sorted(sets), 2):
        shared = sets[a] & sets[b]
        if shared:
            overlaps[(a, b)] = shared
    return overlaps


def assert_disjoint(items_by_split: Mapping[str, Iterable[str]], what: str = "ID") -> None:
    """Raise ``ValueError`` if any item appears in more than one split."""
    overlaps = find_overlaps(items_by_split)
    if overlaps:
        detail = "; ".join(
            f"{a}/{b}: {len(s)} (e.g. {sorted(s)[:3]})" for (a, b), s in overlaps.items()
        )
        raise ValueError(f"{what} appears in more than one split: {detail}")
