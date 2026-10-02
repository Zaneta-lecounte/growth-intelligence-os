"""Markdown rendering helpers and the Narrative type for LLM-written text."""
from __future__ import annotations

import re
from typing import Annotated, Any, Iterable, Sequence

from pydantic import AfterValidator

# A number that is not part of a word: "11%", "$4,286", "35", "2x". Digits inside names such as
# GA4, B2B or Q4 are allowed.
_QUANTITY = re.compile(r"(?<![A-Za-z])\d")


def _no_digits(value: str) -> str:
    if _QUANTITY.search(value):
        raise ValueError(
            "narrative text must not contain numbers; refer to metrics by name and let the "
            "app render computed values"
        )
    return value


# LLM-written prose. Numbers are rendered by code next to it, never written by the LLM.
Narrative = Annotated[str, AfterValidator(_no_digits)]


def md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]], align: Sequence[str] = ()) -> str:
    """GitHub-flavored table. `align` entries: 'l' or 'r' per column (default 'l')."""
    seps = ["---:" if (align[i] if i < len(align) else "l") == "r" else "---" for i in range(len(headers))]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(seps) + "|"]
    lines += ["| " + " | ".join(md_escape(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def md_bullets(items: Iterable[str], empty: str = "_None identified._") -> str:
    items = [i for i in items if i]
    return "\n".join(f"- {i}" for i in items) if items else empty


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def money(x: float) -> str:
    return f"${x:,.0f}"


def num(x: float) -> str:
    return f"{x:,.0f}"
