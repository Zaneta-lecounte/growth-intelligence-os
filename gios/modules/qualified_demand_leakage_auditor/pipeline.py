"""Qualified Demand Leakage Auditor pipeline: analyze (code) -> narrate (LLM, prose only)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from gios.core import data, stats
from gios.core.llm import complete_json
from gios.core.schemas import Signal
from gios.modules.qualified_demand_leakage_auditor import analysis
from gios.modules.qualified_demand_leakage_auditor.analysis import LeakageAnalysis, Settings
from gios.modules.qualified_demand_leakage_auditor.models import LeakageNarrative

NARRATE_PROMPT = "narrate_demand_leakage"


@dataclass
class LeakageResult:
    analysis: LeakageAnalysis
    narrative: Optional[LeakageNarrative]
    narrative_note: str = ""


def leak_id(leak: pd.Series) -> str:
    return f"{leak.source}|{leak.stage}"


def narrative_payload(a: LeakageAnalysis) -> dict[str, str]:
    top = a.top_leak
    top_leaks = [{"leak_id": leak_id(r), "source": r.source, "stage": r.stage,
                  "rate": f"{r.rate:.1%}", "benchmark": f"{r.benchmark:.1%}", "owner": r.owner,
                  "extra_wins_per_month": f"{r.extra_wins_per_month:.1f}"}
                 for _, r in a.leaks.head(5).iterrows()]
    facts = {
        "highest_value_leak_id": leak_id(top) if top is not None else "none",
        "top_leaks": top_leaks,
        "flags": [{"type": f.label, "source": f.source, "owner": f.owner, "evidence": f.evidence} for f in a.flags],
        "owners": a.owners(),
    }
    if top is not None and top.source in a.rejections.index:
        r = a.rejections.loc[top.source]
        facts["top_source_rejection_reasons"] = {
            k: f"{v / r.rejections:.0%}" for k, v in r.drop(["rejections", "qualification_share"]).items() if v}
    return {"facts": json.dumps(facts, indent=2), "leak_id": facts["highest_value_leak_id"]}


def narrate(a: LeakageAnalysis, client: Any = None) -> tuple[Optional[LeakageNarrative], str]:
    if a.top_leak is None:
        return None, "No leak cleared the thresholds, so there is nothing to narrate."
    payload = narrative_payload(a)
    narrative = complete_json(NARRATE_PROMPT, LeakageNarrative, payload, client=client)
    if narrative.about_leak != payload["leak_id"]:
        return None, (f"The narrative on file describes `{narrative.about_leak}`, not the current highest-value "
                      f"leak `{payload['leak_id']}` (in demo mode only the default settings have a cached "
                      "narrative). All numbers below are still current.")
    return narrative, ""


def run(funnel: Optional[pd.DataFrame] = None, sales: Optional[pd.DataFrame] = None,
        settings: Settings = Settings(), client: Any = None) -> LeakageResult:
    funnel = data.load("funnel_by_source") if funnel is None else funnel
    sales = data.load("sales_feedback") if sales is None else sales
    a = analysis.analyze(funnel, sales, settings)
    narrative, note = narrate(a, client)
    return LeakageResult(a, narrative, note)


FLAG_SIGNAL_TYPE = {"high_volume_low_quality": "acquisition", "strong_top_weak_downstream": "funnel",
                    "sla_loss": "operational", "routing_loss": "operational", "qualification_mismatch": "funnel"}


def to_signals(result: LeakageResult) -> list[Signal]:
    a = result.analysis
    signals = []
    revenue = a.leaks.extra_revenue_per_month
    edges = [revenue.max() * q for q in (0.05, 0.15, 0.35, 0.7)] if len(revenue) else [1, 2, 3, 4]
    for _, r in a.leaks.iterrows():
        signals.append(Signal(
            id=f"qdl-leak-{r.source}-{r.stage.replace('→', '-to-').lower()}",
            type="revenue" if r.stage == "Opp→Win" else "funnel",
            evidence_status="observed",
            source="funnel_by_source.csv",
            segment="all",
            journey_stage=analysis.STAGE_JOURNEY[r.stage],
            summary=(f"{r.source} {r.stage} {r.rate:.1%} [{r.ci_low:.1%}–{r.ci_high:.1%}] vs {r.benchmark:.1%} "
                     f"benchmark; closing {a.settings.gap_closure:.0%} of the gap ≈ +{r.extra_wins_per_month:.1f} "
                     f"wins (${r.extra_revenue_per_month:,.0f}) per month. Owner: {analysis.OWNERS[r.owner]}."),
            strength=stats.score_to_strength(r.extra_revenue_per_month, edges),
        ))
    for f in a.flags:
        signals.append(Signal(
            id=f"qdl-flag-{f.kind}-{f.source}",
            type=FLAG_SIGNAL_TYPE[f.kind],
            evidence_status="observed",
            source="sales_feedback.csv" if f.kind in ("sla_loss", "routing_loss", "qualification_mismatch")
            else "funnel_by_source.csv",
            segment="all",
            journey_stage="evaluation",
            summary=f"{f.label} ({f.source}): {f.evidence}. Owner: {analysis.OWNERS[f.owner]}.",
            strength=4 if f.source == (a.top_leak.source if a.top_leak is not None else None) else 3,
        ))
    return signals
