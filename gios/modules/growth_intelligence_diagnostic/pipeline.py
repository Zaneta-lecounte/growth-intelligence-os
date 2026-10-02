"""Growth Intelligence Diagnostic pipeline.

Step 1 (business signal) gates everything. Steps 2–4 are deterministic; Step 5–7 narrative
comes from the LLM with every evidence claim tied to a stored Signal id."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from gios.core import data
from gios.core.llm import complete_json
from gios.core.schemas import Hypothesis, Recommendation, Signal
from gios.modules.growth_intelligence_diagnostic import analysis
from gios.modules.growth_intelligence_diagnostic.analysis import LeakagePoint, RankedHypothesis
from gios.modules.growth_intelligence_diagnostic.models import (
    BusinessSignal,
    DiagnosisOutput,
    checked_output_model,
)

PROMPT = "run_growth_diagnostic"


@dataclass
class Filters:
    channel: Optional[str] = None
    segment: Optional[str] = None
    start: str = "2026-03"
    end: str = "2026-08"

    @property
    def scope_key(self) -> str:
        """Demo-cache key. Cached diagnoses exist per channel scope."""
        return self.channel or "all"


@dataclass
class DiagnosisResult:
    business: BusinessSignal
    filters: Filters
    signals: list[Signal]
    leakage_point: Optional[LeakagePoint]
    output: DiagnosisOutput
    ranked: list[RankedHypothesis] = field(default_factory=list)

    @property
    def by_id(self) -> dict[str, Signal]:
        return {s.id: s for s in self.signals}

    @property
    def top(self) -> RankedHypothesis:
        return self.ranked[0]


def payload(business: BusinessSignal, signals: list[Signal], point: Optional[LeakagePoint]) -> dict[str, str]:
    return {
        "business_signal": business.model_dump_json(indent=2),
        "leakage_point": point.describe() if point else "No stage is significantly below benchmark in this scope.",
        "signals": json.dumps([{"id": s.id, "type": s.type, "evidence_status": s.evidence_status,
                                "channel": s.channel or "any", "segment": s.segment, "strength": s.strength,
                                "summary": s.summary} for s in signals], indent=2),
    }


def diagnose(business: BusinessSignal, filters: Filters, signals: list[Signal], point: Optional[LeakagePoint],
             client: Any = None) -> DiagnosisResult:
    if not signals:
        raise ValueError("No signals in scope. Run the signal-layer modules and save their signals first.")
    model = checked_output_model(frozenset(s.id for s in signals))
    output = complete_json(PROMPT, model, payload(business, signals, point), demo_key=filters.scope_key,
                           client=client)
    by_id = {s.id: s for s in signals}
    return DiagnosisResult(business, filters, signals, point, output,
                           analysis.rank_hypotheses(output.hypotheses, by_id))


def run(business: BusinessSignal, filters: Filters, all_signals: list[Signal], client: Any = None) -> DiagnosisResult:
    signals = analysis.filter_signals(all_signals, filters.channel, filters.segment, filters.start, filters.end)
    point = analysis.leakage_point(data.load("funnel_by_source"), data.load("sales_feedback"),
                                   filters.channel, filters.segment, filters.start, filters.end)
    return diagnose(business, filters, signals, point, client)


def to_hypotheses(result: DiagnosisResult) -> list[Hypothesis]:
    out = []
    for r in result.ranked:
        h = r.root_cause
        out.append(Hypothesis(
            id=f"gid-{result.filters.scope_key}-{r.rank}",
            title=h.title,
            observed_problem=h.observed_problem,
            affected_audience=h.affected_audience,
            causal_explanation=h.causal_explanation,
            intervention=h.intervention,
            expected_behavior_change=h.expected_behavior_change,
            expected_business_outcome=h.expected_business_outcome,
            signal_ids=list(dict.fromkeys(c.signal_id for c in h.supporting)),
            contradicting_signal_ids=list(dict.fromkeys(c.signal_id for c in h.contradicting)),
            missing_evidence=list(h.missing),
            confidence=r.confidence,
        ))
    return out


def save(result: DiagnosisResult, store) -> list[str]:
    """Replace this scope's hypotheses; keep validator scores for hypotheses whose content is unchanged."""
    prefix = f"gid-{result.filters.scope_key}-"
    existing = {h.id: h for h in store.list(Hypothesis, module=analysis.MODULE) if h.id.startswith(prefix)}
    new = to_hypotheses(result)
    for h in new:
        old = existing.get(h.id)
        if old and old.fingerprint() == h.fingerprint():
            for f in Hypothesis.SCORE_FIELDS + ("score_justifications", "validator_recommendation"):
                setattr(h, f, getattr(old, f))
    for stale in set(existing) - {h.id for h in new}:
        store.delete(Hypothesis, stale)
    rec = to_recommendation(result)
    return store.save_many(new, module=analysis.MODULE) + [store.save(rec, module=analysis.MODULE)]


def to_recommendation(result: DiagnosisResult) -> Recommendation:
    o = result.output
    return Recommendation(
        id=f"gid-{result.filters.scope_key}-action",
        title=result.top.root_cause.title,
        action_type=o.next_best_action,
        rationale=o.action_detail,
        primary_metric=o.primary_metric,
        guardrail_metric=o.guardrail_metric,
        downstream_metric=o.downstream_metric,
        confidence=result.top.confidence,
    )
