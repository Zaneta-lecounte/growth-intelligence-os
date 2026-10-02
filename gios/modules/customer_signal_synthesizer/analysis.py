"""Deterministic parts of the Customer Signal Synthesizer.

The LLM only tags text and proposes judgement scores. Counting, frequency bins, trend,
dominant segment/stage, ranking, and Signal creation all happen here.
"""
from __future__ import annotations

import json
from typing import Iterable, Optional

import pandas as pd

from gios.core import stats
from gios.core.schemas import Signal
from gios.modules.customer_signal_synthesizer.models import (
    THEME_LABELS,
    EvidenceTag,
    ThemeAssessment,
)

MODULE = "customer_signal_synthesizer"
BATCH_SIZE = 40
SCORE_COLUMNS = ["severity", "commercial_relevance", "journey_relevance"]
DEFAULT_SCORE = 3  # used when the LLM did not assess a theme; flagged for review
DOMINANT_SHARE = 0.65
STRENGTH_EDGES = (20, 60, 150, 300)  # overall F×S×C×J (1–625) -> signal strength 1–5


def unique_verbatims(evidence: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct verbatim text, with the metadata contexts it appeared in."""
    grouped = evidence.groupby("text_key", sort=True)
    return pd.DataFrame({
        "key": grouped.size().index,
        "verbatim": grouped.verbatim.first().values,
        "mentions": grouped.size().values,
        "sources": grouped.source.agg(lambda s: sorted(set(s))).values,
        "segments": grouped.segment.agg(lambda s: sorted(set(s))).values,
        "journey_stages": grouped.journey_stage.agg(lambda s: sorted(set(s))).values,
    })


def batches(unique: pd.DataFrame, size: int = BATCH_SIZE) -> list[pd.DataFrame]:
    return [unique.iloc[i:i + size] for i in range(0, len(unique), size)]


def batch_payload(batch: pd.DataFrame) -> str:
    """JSON lines for the tagging prompt (metadata given as context, not to be copied)."""
    return "\n".join(
        json.dumps({"key": r.key, "verbatim": r.verbatim, "seen_in_sources": r.sources,
                    "metadata_segments": r.segments, "metadata_journey_stages": r.journey_stages})
        for r in batch.itertuples()
    )


def merge_tags(tags: Iterable[EvidenceTag], wanted_keys: Optional[set[str]] = None) -> dict[str, EvidenceTag]:
    """Keep the first tag per key, optionally restricted to the keys that were asked for."""
    merged: dict[str, EvidenceTag] = {}
    for tag in tags:
        if (wanted_keys is None or tag.key in wanted_keys) and tag.key not in merged:
            merged[tag.key] = tag
    return merged


def apply_tags(evidence: pd.DataFrame, tags: dict[str, EvidenceTag]) -> pd.DataFrame:
    """Attach tags to every evidence row. Source metadata wins for segment/stage; the
    LLM's text-implied values are kept for disagreement checks."""
    out = evidence.copy()
    lookup = {k: t.model_dump() for k, t in tags.items()}
    for field in ["theme", "topic", "sentiment", "business_relevance"]:
        out[field] = out.text_key.map(lambda k: lookup.get(k, {}).get(field))
    out["llm_segment"] = out.text_key.map(lambda k: lookup.get(k, {}).get("segment"))
    out["llm_journey_stage"] = out.text_key.map(lambda k: lookup.get(k, {}).get("journey_stage"))
    out["tagged"] = out.theme.notna()
    return out


PRE_PURCHASE = {"awareness", "consideration", "evaluation", "purchase"}


def _phase(stage: pd.Series) -> pd.Series:
    return stage.map(lambda s: "unknown" if s == "unknown" else "pre" if s in PRE_PURCHASE else "post")


def disagreements(tagged: pd.DataFrame) -> dict[str, int]:
    """Count material disagreements between text-implied and metadata segment / journey phase
    (pre- vs post-purchase; neighbouring stages are not counted)."""
    t = tagged[tagged.tagged]
    seg = (t.llm_segment != "unknown") & (t.llm_segment != t.segment)
    llm_phase, meta_phase = _phase(t.llm_journey_stage), _phase(t.journey_stage)
    stage = (llm_phase != "unknown") & (llm_phase != meta_phase)
    return {"segment": int(seg.sum()), "journey_stage": int(stage.sum())}


def _trend(monthly: pd.Series, months: list[str]) -> tuple[float, str]:
    counts = monthly.reindex(months, fill_value=0)
    early, late = counts.iloc[:2].sum(), counts.iloc[-2:].sum()
    ratio = late / early if early else float("inf") if late else 1.0
    if counts.sum() >= 10 and ratio >= 1.5:
        return ratio, "rising"
    if counts.sum() >= 10 and ratio <= 0.67:
        return ratio, "falling"
    return ratio, "stable"


def _dominant(series: pd.Series) -> tuple[str, str]:
    """(value for Signal, display text). Value is 'all' when nothing dominates."""
    shares = series.value_counts(normalize=True)
    top, share = shares.index[0], shares.iloc[0]
    if share >= DOMINANT_SHARE:
        return top, f"{top} ({share * 100:.0f}%)"
    parts = ", ".join(f"{k} {v * 100:.0f}%" for k, v in shares.head(2).items())
    return "all", f"mixed ({parts})"


RELEVANCE_ORDER = {"high": 0, "medium": 1, "low": 2}


def representative_quote(group: pd.DataFrame) -> str:
    """Most business-relevant verbatim, then most frequent, then alphabetical."""
    counts = group.groupby(["verbatim", "business_relevance"]).size().reset_index(name="n")
    counts["rel"] = counts.business_relevance.map(RELEVANCE_ORDER)
    return counts.sort_values(["rel", "n", "verbatim"], ascending=[True, False, True]).verbatim.iloc[0]


def theme_table(tagged: pd.DataFrame) -> pd.DataFrame:
    """Counted frequency per theme plus deterministic descriptors."""
    t = tagged[tagged.tagged]
    total = len(t)
    months = sorted(tagged.month.unique())
    rows = []
    for theme, g in t.groupby("theme"):
        seg_value, seg_text = _dominant(g.segment)
        stage_value, stage_text = _dominant(g.journey_stage)
        ratio, trend = _trend(g.groupby("month").size(), months)
        quote = representative_quote(g)
        rows.append({
            "theme": theme,
            "label": THEME_LABELS[theme],
            "count": len(g),
            "share": len(g) / total,
            "frequency": stats.frequency_score(len(g), total),
            "segment": seg_value,
            "segment_text": seg_text,
            "journey_stage": g.journey_stage.value_counts().index[0],
            "journey_stage_text": stage_text,
            "trend": trend,
            "trend_ratio": ratio,
            "negative_share": (g.sentiment == "negative").mean(),
            "high_relevance_share": (g.business_relevance == "high").mean(),
            "sources": ", ".join(g.source.value_counts().index[:3]),
            "quote": quote,
        })
    return pd.DataFrame(rows).sort_values(["count", "theme"], ascending=[False, True]).reset_index(drop=True)


def attach_scores(themes: pd.DataFrame, assessments: Iterable[ThemeAssessment]) -> pd.DataFrame:
    """Add LLM-proposed judgement scores (editable later); missing themes get DEFAULT_SCORE."""
    by_theme = {a.theme: a for a in assessments}
    out = themes.copy()
    for col in SCORE_COLUMNS:
        out[col] = out.theme.map(lambda t: getattr(by_theme[t], col) if t in by_theme else DEFAULT_SCORE)
    out["assessed"] = out.theme.isin(by_theme)
    return rank(out)


def rank(themes: pd.DataFrame) -> pd.DataFrame:
    """Overall = Frequency × Severity × Commercial relevance × Journey relevance (1–625)."""
    out = themes.copy()
    for col in ["frequency"] + SCORE_COLUMNS:
        out[col] = out[col].astype(int).clip(1, 5)
    out["severity_x_commercial"] = out.severity * out.commercial_relevance
    out["overall"] = out.frequency * out.severity * out.commercial_relevance * out.journey_relevance
    out = out.sort_values(["overall", "count", "theme"], ascending=[False, False, True]).reset_index(drop=True)
    out["overall_rank"] = range(1, len(out) + 1)
    out["frequency_rank"] = out["count"].rank(ascending=False, method="min").astype(int)
    return out


def period_of(months: pd.Series) -> str:
    return f"{months.min()}..{months.max()}"


def to_signals(themes: pd.DataFrame, period: Optional[str] = None) -> list[Signal]:
    signals = []
    for r in themes.itertuples():
        signals.append(Signal(
            id=f"css-{r.theme}",
            type="customer",
            evidence_status="observed",
            source="customer_evidence.csv",
            segment=r.segment if r.segment in ("smb", "mid_market") else "all",
            journey_stage=r.journey_stage,
            summary=(
                f"{r.label}: {r.count} verbatims ({r.share * 100:.0f}% of evidence), trend {r.trend}; "
                f"F{r.frequency} × S{r.severity} × C{r.commercial_relevance} × J{r.journey_relevance} "
                f"= {r.overall} (rank {r.overall_rank}). e.g. \"{r.quote}\""
            ),
            strength=stats.score_to_strength(r.overall, STRENGTH_EDGES),
            period=period,
            attributes={"theme": r.theme},
        ))
    return signals
