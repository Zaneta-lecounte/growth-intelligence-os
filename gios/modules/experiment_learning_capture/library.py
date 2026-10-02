"""Experiment Library search and filters (deterministic)."""
from __future__ import annotations

from typing import Iterable, Optional

from gios.core.schemas import Learning

LOSS_INTERPRETATIONS = {"negative", "inconclusive"}
SEARCH_FIELDS = ["experiment_name", "hypothesis_text", "original_problem", "what_happened", "learned_about_customer",
                 "learned_about_journey", "learned_about_business", "should_not_conclude", "reusable_principle",
                 "next_hypothesis", "theme", "page", "segment"]


def is_loss_or_inconclusive(learning: Learning) -> bool:
    return learning.interpretation in LOSS_INTERPRETATIONS or learning.decision == "stop"


def search(learnings: Iterable[Learning], query: str = "", themes: Optional[Iterable[str]] = None,
           pages: Optional[Iterable[str]] = None, segments: Optional[Iterable[str]] = None,
           decisions: Optional[Iterable[str]] = None, losses_only: bool = False) -> list[Learning]:
    """All terms in `query` must appear (case-insensitive) somewhere in the learning; facet filters
    are ANDed; empty filters match everything."""
    terms = [t for t in query.lower().split() if t]
    out = []
    for item in learnings:
        haystack = " ".join(str(getattr(item, f) or "") for f in SEARCH_FIELDS).lower()
        if terms and not all(t in haystack for t in terms):
            continue
        if themes and item.theme not in set(themes):
            continue
        if pages and item.page not in set(pages):
            continue
        if segments and item.segment not in set(segments):
            continue
        if decisions and item.decision not in set(decisions):
            continue
        if losses_only and not is_loss_or_inconclusive(item):
            continue
        out.append(item)
    return sorted(out, key=lambda x: x.experiment_id or "")


def facets(learnings: Iterable[Learning]) -> dict[str, list[str]]:
    items = list(learnings)
    return {f: sorted({getattr(x, f) for x in items if getattr(x, f)}) for f in ("theme", "page", "segment", "decision")}
