"""Growth Council Insight Brief: gather the period's inputs (code) -> LLM writes the 10 sections
following Signal → Why it matters → Evidence → Hypothesis → Action → Owner → Learning, citing only
records in the inputs -> owner and exports (code)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from gios.core.llm import complete_json
from gios.core.store import Store
from gios.modules.growth_council_insight_brief import analysis
from gios.modules.growth_council_insight_brief.analysis import BriefInputs
from gios.modules.growth_council_insight_brief.models import BriefDraft, checked_brief_model
from gios.modules.qualified_demand_leakage_auditor.analysis import OWNERS

PROMPT = "write_growth_council_brief"
FULL_PERIOD = ("2026-03", "2026-08")


def demo_key(start: str, end: str) -> str:
    return "full" if (start, end) == FULL_PERIOD else f"{start}_{end}"


def payload(inputs: BriefInputs) -> dict[str, str]:
    def items(xs, fields):
        return [{f: (getattr(x, f) if not callable(getattr(x, f)) else None) for f in fields} for x in xs]

    data = {
        "signals": [{"id": s.id, "type": s.type, "evidence_status": s.evidence_status, "channel": s.channel,
                     "summary": s.summary} for s in inputs.signals],
        "hypotheses": [{"id": h.id, "title": h.label, "confidence": h.confidence, "validator_total": h.validation_total,
                        "validator_recommendation": h.validator_recommendation} for h in inputs.hypotheses],
        "roadmap": [{"id": o.id, "title": o.title, "horizon": o.horizon, "category": o.category,
                     "dependencies": o.dependencies, "recommendation": o.final_recommendation,
                     "primary_metric": o.primary_metric} for o in inputs.roadmap],
        "experiments": [{"id": e.id, "name": e.name, "interpretation": e.interpretation,
                         "recommendation": e.impact_recommendation} for e in inputs.experiments],
        "learnings": [{"id": x.id, "experiment": x.experiment_name, "decision": x.decision,
                       "reusable_principle": x.reusable_principle, "should_not_conclude": x.should_not_conclude}
                      for x in inputs.learnings],
        "owner_classes": OWNERS,
    }
    return {"period": f"{inputs.start} to {inputs.end}", "inputs": json.dumps(data, indent=2, default=str)}


@dataclass
class Brief:
    inputs: BriefInputs
    draft: BriefDraft
    owner: str
    owner_name: str = ""
    suggestions: list[str] = field(default_factory=list)

    @property
    def period(self) -> str:
        return f"{self.inputs.start} to {self.inputs.end}"

    @property
    def owner_label(self) -> str:
        label = OWNERS[self.owner]
        return f"{label} ({self.owner_name})" if self.owner_name else label


def write(inputs: BriefInputs, client: Any = None) -> Brief:
    if not inputs.signals:
        raise ValueError("No signals in this period. Run the signal-layer modules first.")
    model = checked_brief_model(inputs.allowed_ids())
    draft = complete_json(PROMPT, model, payload(inputs), demo_key=demo_key(inputs.start, inputs.end), client=client)
    suggestions = analysis.owner_suggestions(draft.cited_ids(), inputs)
    return Brief(inputs, draft, draft.owner, suggestions=suggestions)


def run(store: Store, start: str, end: str, client: Any = None) -> Brief:
    return write(analysis.gather(store, start, end), client)


def slack(brief: Brief) -> str:
    d = brief.draft
    return analysis.slack_summary(brief.period, d.what_changed, d.why_it_matters, d.working_hypothesis,
                                  d.recommended_action, brief.owner_label, d.measurement, d.decision_needed)
