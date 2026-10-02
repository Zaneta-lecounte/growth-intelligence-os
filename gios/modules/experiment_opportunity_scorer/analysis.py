"""Deterministic parts of the Experiment Opportunity Scorer (experiment-opportunity-scorer.md).

Prefills for R, O, W, H from linked evidence; the GROWTH priority score (on Opportunity);
the sample-size risk check; recommendation rules; key-risk text; category prefill.
Every default is editable in the UI — the score supports judgment, it does not replace it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from gios.core import stats
from gios.core.schemas import Hypothesis, Opportunity, Signal

MODULE = "experiment_opportunity_scorer"
GROWTH = ["growth_impact", "research_evidence", "opportunity_size", "web_evidence", "technical_effort",
          "hypothesis_confidence"]
GROWTH_LETTERS = dict(zip(GROWTH, "GROWTH"))
RISK_LABELS = {
    "strategic_dependency": "Strategic dependency", "measurement_readiness": "Measurement readiness",
    "legal_privacy": "Legal / privacy", "brand": "Brand", "sample_size": "Sample size",
    "implementation_dependency": "Implementation dependency",
}
RECOMMENDATION_LABELS = {"run_now": "Run now", "research_first": "Research first",
                         "instrument_first": "Instrument first", "defer": "Defer", "reject": "Reject"}
QUANTITATIVE = {"funnel", "acquisition", "revenue", "operational"}
JUDGMENT_DEFAULT = 3


@dataclass(frozen=True)
class Settings:
    defer_threshold: float = 20.0   # priority score below which a runnable test is deferred
    max_weeks: float = 6.0          # sample must be reachable within this many weeks
    alpha: float = 0.05
    power: float = 0.80


# --- Prefills -------------------------------------------------------------------------------

# Validator total (0–12) -> H (1–5), aligned with the validator bands:
# 0–4 "do not test" -> 1–2, 5–8 "research / instrument first" -> 3, 9–12 "test-ready" -> 4–5.
H_FROM_VALIDATOR = [(2, 1), (4, 2), (8, 3), (10, 4), (12, 5)]
H_FROM_CONFIDENCE = {"low": 2, "medium": 3, "high": 4}


def h_from_validator(total: int) -> int:
    if not 0 <= total <= 12:
        raise ValueError("validator total must be 0–12")
    return next(h for upper, h in H_FROM_VALIDATOR if total <= upper)


def prefill(h: Optional[Hypothesis], linked: list[Signal]) -> tuple[dict[str, int], dict[str, str]]:
    """Default GROWTH scores and a note explaining each one."""
    by_type: dict[str, list[Signal]] = {}
    for s in linked:
        by_type.setdefault(s.type, []).append(s)
    strongest = lambda types: max((s.strength for t in types for s in by_type.get(t, [])), default=None)  # noqa: E731
    scores, notes = {}, {}

    scores["growth_impact"], notes["growth_impact"] = JUDGMENT_DEFAULT, "judgment input (default 3)"

    r = strongest(["customer"])
    scores["research_evidence"] = r or 1
    notes["research_evidence"] = f"strongest linked customer signal ({r})" if r else "no customer signal linked"

    sized = [s.strength for s in linked if s.id.startswith(("qdl-leak-", "bfa-"))]
    scores["opportunity_size"] = max(sized) if sized else JUDGMENT_DEFAULT
    notes["opportunity_size"] = (f"strongest sized leak / friction signal ({max(sized)})" if sized
                                 else "no sized signal linked (default 3)")

    w = strongest(["behavioral"])
    if w:
        scores["web_evidence"], notes["web_evidence"] = w, f"strongest linked behavioral signal ({w})"
    elif by_type.keys() & QUANTITATIVE:
        scores["web_evidence"], notes["web_evidence"] = 2, "only indirect quantitative (funnel) signals"
    else:
        scores["web_evidence"], notes["web_evidence"] = 1, "no behavioral or quantitative signal linked"

    scores["technical_effort"], notes["technical_effort"] = JUDGMENT_DEFAULT, "judgment input (default 3)"

    if h is not None and h.validation_total is not None:
        scores["hypothesis_confidence"] = h_from_validator(h.validation_total)
        notes["hypothesis_confidence"] = f"validator total {h.validation_total}/12"
    elif h is not None and h.confidence:
        scores["hypothesis_confidence"] = H_FROM_CONFIDENCE[h.confidence]
        notes["hypothesis_confidence"] = f"not validated; diagnostic confidence {h.confidence}"
    else:
        scores["hypothesis_confidence"] = JUDGMENT_DEFAULT
        notes["hypothesis_confidence"] = "not validated (default 3)"
    return scores, notes


# --- Signal id parsing (ids are created deterministically by the Phase 1 modules) -------------

STAGE_SLUGS = {"visit-to-lead": "Visit→Lead", "lead-to-mql": "Lead→MQL", "mql-to-sql": "MQL→SQL",
               "sql-to-opp": "SQL→Opp", "opp-to-win": "Opp→Win"}
FLAG_STAGE = {"qualification_mismatch": "MQL→SQL", "strong_top_weak_downstream": "MQL→SQL",
              "sla_loss": "MQL→SQL", "high_volume_low_quality": "Lead→MQL", "rising_cac": "Visit→Lead"}


def parse_friction_id(sid: str) -> Optional[dict]:
    """bfa-{page}-{device|source}-{value} -> {page, device|channel}."""
    parts = sid.split("-", 3)
    if len(parts) != 4 or parts[0] != "bfa" or parts[2] not in ("device", "source"):
        return None
    return {"page": parts[1], ("device" if parts[2] == "device" else "channel"): parts[3]}


def parse_leak_id(sid: str) -> Optional[dict]:
    if not sid.startswith("qdl-leak-"):
        return None
    rest = sid[len("qdl-leak-"):]
    for slug, stage in STAGE_SLUGS.items():
        if rest.endswith("-" + slug):
            return {"channel": rest[: -len(slug) - 1], "funnel_stage": stage}
    return None


def parse_flag_id(sid: str) -> Optional[dict]:
    if not sid.startswith("qdl-flag-"):
        return None
    rest = sid[len("qdl-flag-"):]
    for kind, stage in FLAG_STAGE.items():
        if rest.startswith(kind + "-"):
            source = rest[len(kind) + 1:]
            return None if source == "all" else {"channel": source, "funnel_stage": stage}
    return None


def infer_population(signal_ids: list[str]) -> dict:
    """Where an experiment on this opportunity would run: a web slice from linked friction
    signals, else a funnel stage from linked leak or flag signals."""
    for parse, unit in ((parse_friction_id, "web"), (parse_leak_id, "funnel"), (parse_flag_id, "funnel")):
        for sid in signal_ids:
            found = parse(sid)
            if found:
                return {"test_unit": unit, **found}
    return {}


def is_fix(linked: list[Signal]) -> bool:
    return any(s.type == "behavioral" and "Friction: technical" in s.summary for s in linked)


def category_for(linked: list[Signal]) -> str:
    """Spec classification prefill from the most specific linked evidence, in order: on-site
    friction → conversion; a leaking funnel stage; a leakage flag; then the signal types."""
    stage_category = {"Visit→Lead": "conversion", "Lead→MQL": "acquisition", "MQL→SQL": "qualification",
                      "SQL→Opp": "revenue", "Opp→Win": "revenue"}
    ids = [s.id for s in linked]
    if any(parse_friction_id(i) for i in ids):
        return "conversion"
    for sid in ids:
        if leak := parse_leak_id(sid):
            return stage_category[leak["funnel_stage"]]
    for sid in ids:
        if sid.startswith(("qdl-flag-sla_loss", "qdl-flag-routing")):
            return "operations_measurement"
        if sid.startswith(("qdl-flag-rising_cac", "qdl-flag-high_volume_low_quality")):
            return "acquisition"
        if flag := parse_flag_id(sid):
            return stage_category[flag["funnel_stage"]]
    types = {s.type for s in linked}
    if "behavioral" in types:
        return "conversion"
    if "operational" in types:
        return "operations_measurement"
    if "acquisition" in types:
        return "acquisition"
    return "customer_problem"


# --- Sample size ------------------------------------------------------------------------------


@dataclass
class SampleCheck:
    available: bool
    description: str
    baseline: float = 0.0
    weekly_volume: float = 0.0
    n_per_arm: int = 0
    weeks: float = math.inf
    flagged: bool = False

    def describe(self, max_weeks: float) -> str:
        if not self.available:
            return self.description
        weeks = "∞" if math.isinf(self.weeks) else f"{self.weeks:.1f}"
        return (f"{self.description}: baseline {self.baseline:.1%}, ~{self.weekly_volume:,.0f}/week; "
                f"{self.n_per_arm:,} per arm → {weeks} weeks (limit {max_weeks:g})")


def population_stats(o: Opportunity, web: pd.DataFrame, funnel: pd.DataFrame) -> Optional[tuple[float, float, str]]:
    """(baseline rate, weekly eligible volume, description) for the opportunity's test population."""
    if o.test_unit == "web" and o.page:
        last = web.date.max()
        recent = web[(web.date > (pd.Timestamp(last) - pd.Timedelta(days=28)).strftime("%Y-%m-%d"))]
        s = recent[recent.page == o.page]
        if o.channel:
            s = s[s.source == o.channel]
        if o.device:
            s = s[s.device == o.device]
        num = "form_completes" if o.page == "demo" else "cta_clicks"
        what = "demo-form completion per session" if o.page == "demo" else "CTA clicks per session"
        who = " · ".join(x for x in [o.page + " page", o.channel, o.device] if x)
        return stats.safe_rate(s[num].sum(), s.sessions.sum()), s.sessions.sum() / 4, f"{who} ({what})"
    if o.test_unit == "funnel" and o.channel and o.funnel_stage:
        from gios.modules.qualified_demand_leakage_auditor.analysis import STAGES

        den, num = next((d, n) for d, n, name in STAGES if name == o.funnel_stage)
        last_month = funnel[(funnel.month == funnel.month.max()) & (funnel.source == o.channel)]
        weekly = last_month[den].sum() * 12 / 52
        return (stats.safe_rate(last_month[num].sum(), last_month[den].sum()), weekly,
                f"{o.channel} {o.funnel_stage}")
    return None


def sample_check(o: Opportunity, web: pd.DataFrame, funnel: pd.DataFrame, settings: Settings) -> SampleCheck:
    pop = population_stats(o, web, funnel)
    if pop is None:
        return SampleCheck(False, "No test population inferred; set it manually to check sample size.")
    baseline, weekly, description = pop
    n = stats.sample_size_per_arm(baseline, o.mde, settings.alpha, settings.power)
    if n == 0:
        return SampleCheck(True, description, baseline, weekly, 0, math.inf, True)
    weeks = stats.weeks_to_sample(n, weekly)
    return SampleCheck(True, description, baseline, weekly, n, weeks, weeks > settings.max_weeks)


# --- Recommendation rules -----------------------------------------------------------------------

RULES = [
    "Validator total 0–4 → Reject (not test-ready).",
    "Validator total 5–8 → Instrument first if measurement readiness is flagged or the validator said so, "
    "otherwise Research first, regardless of score.",
    "Not validated → Research first.",
    "Measurement-readiness risk flagged → Instrument first.",
    "Legal / privacy risk flagged → Research first (needs review).",
    "Priority score below the defer threshold → Defer.",
    "Otherwise → Run now. Other risk flags (sample size, dependencies, brand) are reported as the key risk.",
]


def recommend(o: Opportunity, settings: Settings, validator_rec: Optional[str] = None) -> tuple[str, str]:
    """(recommendation, the rule that fired)."""
    t = o.validator_total
    if t is not None and t <= 4:
        return "reject", RULES[0]
    if t is not None and t <= 8:
        instrument = "measurement_readiness" in o.risk_flags or validator_rec == "instrument_first"
        return ("instrument_first" if instrument else "research_first"), RULES[1]
    if t is None:
        return "research_first", RULES[2]
    if "measurement_readiness" in o.risk_flags:
        return "instrument_first", RULES[3]
    if "legal_privacy" in o.risk_flags:
        return "research_first", RULES[4]
    if (o.priority_score or 0) < settings.defer_threshold:
        return "defer", RULES[5]
    return "run_now", RULES[6]


def key_risk(o: Opportunity, check: SampleCheck, settings: Settings) -> str:
    risks = []
    if "sample_size" in o.risk_flags and check.available:
        weeks = "∞" if math.isinf(check.weeks) else f"{check.weeks:.0f}"
        risks.append(f"Sample size: ~{weeks} weeks at {o.mde:.0%} MDE")
    elif "sample_size" in o.risk_flags:
        risks.append("Sample size")
    risks += [RISK_LABELS[f] for f in o.risk_flags if f != "sample_size"]
    return "; ".join(risks) or "None flagged"
