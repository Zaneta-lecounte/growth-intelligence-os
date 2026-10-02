"""Behavioral Friction Analyzer pipeline: detect + size (code) -> classify (LLM) -> evidence check (code)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from gios.core import data
from gios.core.llm import complete_json
from gios.core.schemas import Signal
from gios.modules.behavioral_friction_analyzer import analysis
from gios.modules.behavioral_friction_analyzer.analysis import Finding, Thresholds
from gios.modules.behavioral_friction_analyzer.models import FrictionBatch, FrictionClassification

CLASSIFY_PROMPT = "classify_behavioral_friction"
UNCONFIRMED = "unconfirmed: no customer evidence linked"


@dataclass
class EvidenceCheck:
    linked_signal_ids: list[str]
    dropped_signal_ids: list[str]

    @property
    def status(self) -> str:
        return "supported by customer evidence" if self.linked_signal_ids else UNCONFIRMED


@dataclass
class BehavioralResult:
    findings: list[Finding]
    classifications: dict[str, FrictionClassification]
    evidence: dict[str, EvidenceCheck]
    customer_signals: list[Signal]
    thresholds: Thresholds
    meta: dict[str, Any] = field(default_factory=dict)


def check_evidence(c: FrictionClassification, customer_signals: list[Signal]) -> EvidenceCheck:
    """A cause may only be called supported if it cites customer signals that actually exist."""
    available = {s.id for s in customer_signals}
    linked = [i for i in c.supporting_customer_signal_ids if i in available]
    dropped = [i for i in c.supporting_customer_signal_ids if i not in available]
    return EvidenceCheck(linked, dropped)


def findings_payload(findings: list[Finding]) -> str:
    items = []
    for f in findings:
        item = {"finding_id": f.id, "title": f.title, "page": f.page, f.dimension: f.value,
                "observations": [fl.describe() for fl in f.flags]}
        if "form_start_rate" in f.context:
            item["form_start_rate"] = f"{f.context['form_start_rate']:.1%} vs {f.context['form_start_rate_rest']:.1%} elsewhere"
        if f.sizing.get("sized"):
            item["estimated_impact"] = f"~{f.sizing['lost_leads_per_month']:,.0f} lost leads per month"
        items.append(item)
    return json.dumps(items, indent=2)


def customer_payload(customer_signals: list[Signal]) -> str:
    if not customer_signals:
        return "None available. Do not cite any customer signal ids."
    return json.dumps([{"id": s.id, "summary": s.summary} for s in customer_signals], indent=2)


def classify(findings: list[Finding], customer_signals: list[Signal], client: Any = None) -> dict[str, FrictionClassification]:
    if not findings:
        return {}
    batch = complete_json(CLASSIFY_PROMPT, FrictionBatch, {
        "findings": findings_payload(findings),
        "customer_signals": customer_payload(customer_signals),
    }, client=client)
    wanted = {f.id for f in findings}
    out: dict[str, FrictionClassification] = {}
    for c in batch.classifications:
        if c.finding_id in wanted and c.finding_id not in out:
            out[c.finding_id] = c
    return out


def run(web: Optional[pd.DataFrame] = None, funnel: Optional[pd.DataFrame] = None,
        thresholds: Thresholds = Thresholds(), customer_signals: Optional[list[Signal]] = None,
        client: Any = None) -> BehavioralResult:
    web = data.load("web_behavior") if web is None else web
    funnel = data.load("funnel_by_source") if funnel is None else funnel
    customer_signals = customer_signals or []
    findings = analysis.detect(web, thresholds)
    analysis.size_all(findings, web, funnel)
    classifications = classify(findings, customer_signals, client)
    evidence = {fid: check_evidence(c, customer_signals) for fid, c in classifications.items()}
    return BehavioralResult(findings, classifications, evidence, customer_signals, thresholds,
                            meta={"months": f"{web.month.min()} to {web.month.max()}",
                                  "sessions": int(web.sessions.sum())})


STAGE_BY_PAGE = analysis.PAGE_STAGE


def to_signals(result: BehavioralResult) -> list[Signal]:
    from gios.core import stats

    signals = []
    for f in result.findings:
        c = result.classifications.get(f.id)
        ev = result.evidence.get(f.id)
        if f.sizing.get("sized"):
            strength = stats.score_to_strength(f.sizing["lost_leads_per_month"], (5, 15, 40, 100))
        else:
            strength = min(stats.score_to_strength(f.max_abs_z, (3, 5, 10, 20)), 2)
        friction = c.friction_type if c else "unclassified"
        signals.append(Signal(
            id=f"bfa-{f.id.replace('|', '-').replace('=', '-')}",
            type="behavioral",
            evidence_status="observed",
            source="web_behavior.csv",
            segment="all",
            journey_stage=STAGE_BY_PAGE.get(f.page, "unknown"),
            summary=(f"{f.title}: " + "; ".join(fl.describe() for fl in f.flags)
                     + f". Friction: {friction} ({ev.status if ev else 'not classified'})."),
            strength=strength,
        ))
    return signals
