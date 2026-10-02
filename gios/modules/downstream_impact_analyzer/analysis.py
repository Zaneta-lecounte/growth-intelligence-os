"""Deterministic Downstream Impact Analyzer (downstream-impact-analyzer.md).

Two-proportion tests at every stage (per visitor, so randomization holds), quality rates
between stages, revenue per visitor and CAC, power checks, observation-window check against
lead→win velocity, segment breakdown, and rule-based interpretation and recommendation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import pandas as pd

from gios.core import stats

MODULE = "downstream_impact_analyzer"
COUNTS = ["visitors", "conversions", "mqls", "sqls", "opps", "wins"]
# Per-visitor stages (immediate first, then downstream)
STAGES = [("conversions", "Primary conversion"), ("mqls", "MQL"), ("sqls", "SQL"), ("opps", "Opportunity"),
          ("wins", "Win")]
DOWNSTREAM = ["sqls", "opps", "wins"]
# Quality rates between consecutive stages
QUALITY = [("conversions", "mqls", "Lead→MQL"), ("mqls", "sqls", "MQL→SQL"), ("sqls", "opps", "SQL→Opp"),
           ("opps", "wins", "Opp→Win")]
INTERPRETATION_LABELS = {
    "clear_positive": "Clear positive",
    "top_funnel_positive_downstream_neutral": "Top-funnel positive / downstream neutral",
    "volume_quality_tradeoff": "Volume-quality tradeoff",
    "inconclusive": "Inconclusive",
    "negative": "Negative",
    "needs_longer_observation": "Needs longer observation",
}
RECOMMENDATION_LABELS = {"scale": "Scale", "iterate": "Iterate", "retest": "Retest", "stop": "Stop",
                         "observe_longer": "Observe longer"}
RULES = [
    "Primary conversion significantly worse → Negative.",
    "Primary significantly better but a quality rate between stages significantly worse → Volume-quality tradeoff "
    "(more volume, lower quality; holds even when downstream volume falls).",
    "A downstream per-visitor stage (SQL, opportunity, win) significantly worse → Negative.",
    "Observation window shorter than the traffic's median lead→win velocity → Needs longer observation.",
    "Primary significantly better and a downstream per-visitor stage significantly better → Clear positive.",
    "Primary significantly better, downstream not significantly different → Top-funnel positive / downstream neutral.",
    "Otherwise → Inconclusive.",
]
RECOMMENDATION_RULES = {
    "clear_positive": "scale", "top_funnel_positive_downstream_neutral": "iterate",
    "volume_quality_tradeoff": "iterate", "negative": "stop", "needs_longer_observation": "observe_longer",
}


@dataclass(frozen=True)
class Settings:
    alpha: float = 0.05
    min_events: int = 30   # per arm, below which a stage test is flagged underpowered
    power: float = 0.80
    meaningful_lift: float = 0.10  # smallest relative primary effect worth detecting


@dataclass
class StageResult:
    key: str
    label: str
    control: tuple[int, int]
    treatment: tuple[int, int]
    test: dict
    events_per_arm: int
    mde: float
    underpowered: bool

    def describe(self) -> str:
        t = self.test
        sig = "significant" if t["significant"] else "not significant"
        return (f"{self.label}: {t['p_control']:.2%} → {t['p_treatment']:.2%} ({t['lift']:+.1%} relative; "
                f"diff {t['diff']:+.2%} [{t['ci_low']:+.2%}, {t['ci_high']:+.2%}], p = {t['p_value']:.3f}, {sig})"
                + (f"; underpowered ({self.events_per_arm} events per arm)" if self.underpowered else ""))


def _stage(key: str, label: str, c: tuple[int, int], t: tuple[int, int], s: Settings) -> StageResult:
    test = stats.two_proportion_test(c[0], c[1], t[0], t[1], s.alpha)
    events = min(c[0], t[0])
    mde = stats.mde_relative(test["p_control"], c[1], s.alpha, s.power)
    return StageResult(key, label, c, t, test, events, mde, events < s.min_events)


@dataclass
class Money:
    rpv_control: float
    rpv_treatment: float
    rpv_ci: tuple[float, float]
    cac_control: float
    cac_treatment: float

    @property
    def cac_change(self) -> float:
        if math.isinf(self.cac_control) or math.isinf(self.cac_treatment) or not self.cac_control:
            return math.nan
        return self.cac_treatment / self.cac_control - 1


def money(c: pd.Series, t: pd.Series, win_test: dict) -> Money:
    """Revenue per visitor and CAC per variant. The RPV CI scales the win-rate CI by the pooled
    revenue per win (assumes deal size does not differ between arms)."""
    rpw = stats.safe_rate(c.revenue + t.revenue, c.wins + t.wins)
    cac = lambda v: v.spend / v.wins if v.wins else math.inf  # noqa: E731
    return Money(c.revenue / c.visitors, t.revenue / t.visitors,
                 (win_test["ci_low"] * rpw, win_test["ci_high"] * rpw), cac(c), cac(t))


def velocity_for_mix(cells: Optional[pd.DataFrame], velocity: pd.DataFrame) -> Optional[float]:
    """Traffic-weighted median lead→win days for the experiment's source × segment mix."""
    if cells is None or cells.empty:
        return None
    mix = cells.groupby(["source", "segment"]).visitors.sum()
    v = velocity.set_index(["source", "segment"]).median_days_lead_to_win
    common = mix.index.intersection(v.index)
    if not len(common):
        return None
    return float((mix[common] * v[common]).sum() / mix[common].sum())


@dataclass
class SegmentFinding:
    dimension: str
    value: str
    share: float
    primary: dict
    sql: dict
    interaction_z: float  # segment effect vs the rest of the experiment's effect (primary)

    @property
    def differs(self) -> bool:
        return abs(self.interaction_z) >= 1.96

    def describe(self) -> str:
        sig = lambda t: "significant" if t["significant"] else "n.s."  # noqa: E731
        return (f"{self.dimension} = {self.value} ({self.share:.0%} of traffic): primary {self.primary['lift']:+.0%} "
                f"({sig(self.primary)}), SQL per visitor {self.sql['lift']:+.0%} ({sig(self.sql)})")


def interaction_z(seg: dict, rest: dict) -> float:
    """z for the difference between two independent treatment effects (diff of diffs)."""
    se = lambda t: (t["ci_high"] - t["ci_low"]) / (2 * 1.959964)  # noqa: E731
    denom = math.sqrt(se(seg) ** 2 + se(rest) ** 2)
    return 0.0 if denom == 0 else (seg["diff"] - rest["diff"]) / denom


def segment_breakdown(cells: Optional[pd.DataFrame], control: str, treatment: str,
                      overall_primary: dict, s: Settings) -> tuple[list[SegmentFinding], list[str]]:
    """Per source / device / segment results. A segment is noted as differing only when its
    primary effect differs significantly from the rest of the experiment (interaction test), not
    merely because a smaller slice lost significance."""
    if cells is None or cells.empty:
        return [], []
    findings, notes = [], []
    total = cells.visitors.sum()

    def test(frame, num):
        c = frame[frame.variant == control][COUNTS].sum()
        t = frame[frame.variant == treatment][COUNTS].sum()
        return stats.two_proportion_test(int(c[num]), int(c.visitors), int(t[num]), int(t.visitors), s.alpha)

    for dim in ("source", "device", "segment"):
        if cells[dim].nunique() < 2:
            continue
        for value, g in cells.groupby(dim):
            p = test(g, "conversions")
            rest = test(cells[cells[dim] != value], "conversions")
            f = SegmentFinding(dim, value, g.visitors.sum() / total, p, test(g, "sqls"), interaction_z(p, rest))
            findings.append(f)
            if f.differs:
                notes.append(f"{dim} = {value}: primary {p['lift']:+.0%} vs {rest['lift']:+.0%} in the rest of the "
                             f"traffic (interaction z = {f.interaction_z:+.1f})")
    return findings, notes


@dataclass
class ImpactResult:
    experiment_id: str
    name: str
    control: str
    treatment: str
    stages: list[StageResult]
    quality: list[StageResult]
    money: Money
    window_days: Optional[int]
    velocity_days: Optional[float]
    segments: list[SegmentFinding]
    segment_notes: list[str]
    interpretation: str
    rule: str
    recommendation: str
    settings: Settings
    meta: dict = field(default_factory=dict)

    def stage(self, key: str) -> StageResult:
        return next(s for s in self.stages if s.key == key)

    @property
    def needs_longer(self) -> bool:
        return self.window_days is not None and self.velocity_days is not None and self.window_days < self.velocity_days


def interpret(stages: list[StageResult], quality: list[StageResult], needs_longer: bool) -> tuple[str, str]:
    """Spec step 4, by the documented rules in RULES (applied in order)."""
    by = {s.key: s for s in stages}
    primary = by["conversions"].test
    down_neg = any(by[k].test["significant"] and by[k].test["diff"] < 0 for k in DOWNSTREAM)
    down_pos = any(by[k].test["significant"] and by[k].test["diff"] > 0 for k in DOWNSTREAM)
    quality_neg = any(q.test["significant"] and q.test["diff"] < 0 for q in quality)
    primary_up = primary["significant"] and primary["diff"] > 0
    if primary["significant"] and primary["diff"] < 0:
        return "negative", RULES[0]
    if primary_up and quality_neg:
        return "volume_quality_tradeoff", RULES[1]
    if down_neg:
        return "negative", RULES[2]
    if needs_longer:
        return "needs_longer_observation", RULES[3]
    if primary_up and down_pos:
        return "clear_positive", RULES[4]
    if primary_up:
        return "top_funnel_positive_downstream_neutral", RULES[5]
    return "inconclusive", RULES[6]


def recommend(interpretation: str, primary: StageResult, meaningful_lift: float = 0.10) -> tuple[str, str]:
    """Scale / Iterate / Retest / Stop / Observe longer. An inconclusive test is retested when it
    could not have detected a meaningful effect (primary MDE above `meaningful_lift`), else stopped."""
    if interpretation == "inconclusive":
        if primary.mde > meaningful_lift or primary.underpowered:
            return "retest", (f"Inconclusive, and the test could only detect effects of {primary.mde:.0%} or more "
                              f"(meaningful: {meaningful_lift:.0%}) → Retest with a larger sample.")
        return "stop", (f"Inconclusive although the test could detect a {primary.mde:.0%} effect → Stop: there is "
                        "no effect worth chasing.")
    rec = RECOMMENDATION_RULES[interpretation]
    return rec, f"{INTERPRETATION_LABELS[interpretation]} → {RECOMMENDATION_LABELS[rec]}."


def analyze(experiment_id: str, name: str, variants: pd.DataFrame, cells: Optional[pd.DataFrame] = None,
            velocity_days: Optional[float] = None, start: Optional[date] = None,
            observed_through: Optional[date] = None, settings: Settings = Settings(),
            meta: Optional[dict] = None) -> ImpactResult:
    """`variants`: two rows (control first) with COUNTS + revenue + spend, indexed by variant name."""
    if len(variants) != 2:
        raise ValueError("the analyzer compares exactly two variants (control first)")
    control, treatment = variants.index[0], variants.index[1]
    c, t = variants.loc[control], variants.loc[treatment]
    stages = [_stage(k, label, (int(c[k]), int(c.visitors)), (int(t[k]), int(t.visitors)), settings)
              for k, label in STAGES]
    quality = [_stage(f"{a}_{b}", label, (int(c[b]), int(c[a])), (int(t[b]), int(t[a])), settings)
               for a, b, label in QUALITY]
    window = (observed_through - start).days if start and observed_through else None
    needs_longer = window is not None and velocity_days is not None and window < velocity_days
    interpretation, rule = interpret(stages, quality, needs_longer)
    rec, _ = recommend(interpretation, stages[0], settings.meaningful_lift)
    segs, notes = segment_breakdown(cells, control, treatment, stages[0].test, settings)
    return ImpactResult(experiment_id, name, control, treatment, stages, quality,
                        money(c, t, stages[-1].test), window, velocity_days, segs, notes, interpretation, rule, rec,
                        settings, meta or {})
