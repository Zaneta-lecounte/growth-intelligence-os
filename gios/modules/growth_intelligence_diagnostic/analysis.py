"""Deterministic parts of the Growth Intelligence Diagnostic: signal filtering and
classification, business-signal baselines, the leakage point (from the Leakage Auditor),
hypothesis ranking, and confidence."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from gios.core import stats
from gios.core.schemas import Signal
from gios.modules.growth_intelligence_diagnostic.models import EvidenceClaim, RootCause
from gios.modules.qualified_demand_leakage_auditor import analysis as leakage

MODULE = "growth_intelligence_diagnostic"
SIGNAL_TYPES = ["customer", "behavioral", "funnel", "acquisition", "operational", "revenue"]
# Spec output groups the six types under four evidence headings.
EVIDENCE_GROUPS = {
    "Customer": ["customer"],
    "Behavioral": ["behavioral"],
    "Funnel": ["funnel", "acquisition"],
    "Business / Revenue": ["revenue", "operational"],
}
CONFIDENCE_ORDER = ["low", "medium", "high"]


# --- Signals ---------------------------------------------------------------------------------


def filter_signals(signals: list[Signal], channel: Optional[str] = None, segment: Optional[str] = None,
                   start: Optional[str] = None, end: Optional[str] = None) -> list[Signal]:
    """Channel-agnostic (channel=None) and all-segment signals always apply."""
    out = []
    for s in signals:
        if channel and s.channel not in (channel, None):
            continue
        if segment and s.segment not in (segment, "all"):
            continue
        if start and end and not s.covers_month_range(start, end):
            continue
        out.append(s)
    return out


def classify(signals: list[Signal]) -> dict[str, list[Signal]]:
    by_type = {t: [] for t in SIGNAL_TYPES}
    for s in sorted(signals, key=lambda s: (-s.strength, s.id)):
        by_type[s.type].append(s)
    return by_type


# --- Step 1 helper: business signal from data --------------------------------------------------

METRICS = {
    # name: (numerator, denominator, higher_is_worse, format)
    "CAC": ("spend", "wins", True, "money"),
    "Cost per SQL": ("spend", "sqls", True, "money"),
    "Visit→Lead": ("leads", "visits", False, "pct"),
    "Lead→MQL": ("mqls", "leads", False, "pct"),
    "MQL→SQL": ("sqls", "mqls", False, "pct"),
    "Lead→Win": ("wins", "leads", False, "pct"),
}


def _fmt(value: float, kind: str) -> str:
    return f"${value:,.0f}" if kind == "money" else f"{value:.1%}"


def metric_halves(funnel: pd.DataFrame, metric: str, channel: Optional[str], segment: Optional[str],
                  start: str, end: str) -> dict:
    num, den, worse_high, kind = METRICS[metric]
    f = funnel[(funnel.month >= start) & (funnel.month <= end)]
    if channel:
        f = f[f.source == channel]
    if segment:
        f = f[f.segment == segment]
    months = sorted(f.month.unique())
    half = len(months) // 2
    early, late = months[:half], months[len(months) - half:]
    val = lambda m: stats.safe_rate(f[f.month.isin(m)][num].sum(), f[f.month.isin(m)][den].sum())  # noqa: E731
    v_early, v_late = val(early), val(late)
    change = stats.pct_deviation(v_late, v_early)
    worse = change > 0 if worse_high else change < 0
    return {"metric": metric, "early": v_early, "late": v_late, "change": change, "worse": worse, "kind": kind,
            "early_months": f"{early[0]}..{early[-1]}" if early else "", "late_months": f"{late[0]}..{late[-1]}" if late else ""}


def suggest_business_signal(funnel: pd.DataFrame, metric: str, channel: Optional[str], segment: Optional[str],
                            start: str, end: str) -> dict[str, str]:
    """Deterministic Step 1 draft: the metric in the later half vs the earlier half of the period."""
    h = metric_halves(funnel, metric, channel, segment, start, end)
    who = (channel or "all channels").replace("_", " ")
    direction = "worsened" if h["worse"] else "improved or held"
    return {
        "target_metric": f"{metric} ({who})",
        "what": f"{who.capitalize()} {metric} {direction}: {_fmt(h['late'], h['kind'])} in {h['late_months']} "
                f"({h['change']:+.0%})",
        "baseline": f"{_fmt(h['early'], h['kind'])} in {h['early_months']} (same channel and segment, earlier half "
                    "of the period)",
        "segment": f"{(segment or 'all segments').replace('_', ' ')}, {who}",
        "period": f"{start} to {end}",
    }


# --- Step 4: leakage point (deterministic, from the Leakage Auditor) -------------------------


@dataclass
class LeakagePoint:
    source: str
    stage: str
    rate: float
    ci_low: float
    ci_high: float
    benchmark: float
    volume: int
    extra_wins_per_month: float
    extra_revenue_per_month: float
    owner: str
    gap_closure: float

    def describe(self) -> str:
        return (f"{self.source} {self.stage}: {self.rate:.1%} [{self.ci_low:.1%}–{self.ci_high:.1%}] vs "
                f"{self.benchmark:.1%} benchmark on {self.volume:,}; closing {self.gap_closure:.0%} of the gap ≈ "
                f"+{self.extra_wins_per_month:.1f} wins (${self.extra_revenue_per_month:,.0f}) per month. "
                f"Owner: {leakage.OWNERS[self.owner]}.")


def leakage_point(funnel: pd.DataFrame, sales: pd.DataFrame, channel: Optional[str], segment: Optional[str],
                  start: str, end: str, settings: leakage.Settings = leakage.Settings()) -> Optional[LeakagePoint]:
    f = funnel[(funnel.month >= start) & (funnel.month <= end)]
    if segment:
        f = f[f.segment == segment]
    if f.empty:
        return None
    a = leakage.analyze(f, sales, settings)
    leaks = a.leaks if not channel else a.leaks[a.leaks.source == channel]
    if leaks.empty:
        return None
    r = leaks.iloc[0]
    return LeakagePoint(r.source, r.stage, r.rate, r.ci_low, r.ci_high, r.benchmark, int(r.volume),
                        r.extra_wins_per_month, r.extra_revenue_per_month, r.owner, settings.gap_closure)


# --- Steps 5: ranking and confidence ----------------------------------------------------------


def normalize_claims(claims: list[EvidenceClaim], by_id: dict[str, Signal]) -> list[EvidenceClaim]:
    """A claim can't be more certain than its signal: inferred signals make inferred claims."""
    out = []
    for c in claims:
        status = "inferred" if by_id[c.signal_id].evidence_status != "observed" else c.status
        out.append(c.model_copy(update={"status": status}))
    return out


def evidence_score(h: RootCause, by_id: dict[str, Signal]) -> int:
    sup = {c.signal_id for c in h.supporting}
    con = {c.signal_id for c in h.contradicting}
    return sum(by_id[i].strength for i in sup) - sum(by_id[i].strength for i in con)


def confidence(h: RootCause, by_id: dict[str, Signal]) -> str:
    """High: ≥3 supporting signal types with ≥2 observed. Medium: ≥2 types. Low otherwise.
    Any contradicting evidence lowers confidence one level."""
    types = {by_id[c.signal_id].type for c in h.supporting}
    observed_types = {by_id[c.signal_id].type for c in h.supporting if c.status == "observed"}
    level = 2 if len(types) >= 3 and len(observed_types) >= 2 else 1 if len(types) >= 2 else 0
    if h.contradicting:
        level = max(level - 1, 0)
    return CONFIDENCE_ORDER[level]


@dataclass
class RankedHypothesis:
    rank: int
    root_cause: RootCause
    score: int
    confidence: str
    supporting_types: list[str]


def rank_hypotheses(hypotheses: list[RootCause], by_id: dict[str, Signal]) -> list[RankedHypothesis]:
    normalized = [h.model_copy(update={"supporting": normalize_claims(h.supporting, by_id),
                                       "contradicting": normalize_claims(h.contradicting, by_id)})
                  for h in hypotheses]
    scored = sorted(enumerate(normalized), key=lambda ih: (-evidence_score(ih[1], by_id), ih[0]))
    return [RankedHypothesis(rank, h, evidence_score(h, by_id), confidence(h, by_id),
                             sorted({by_id[c.signal_id].type for c in h.supporting}, key=SIGNAL_TYPES.index))
            for rank, (_, h) in enumerate(scored, 1)]
