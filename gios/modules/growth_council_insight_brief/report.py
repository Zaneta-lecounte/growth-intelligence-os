"""Render the Growth Council Insight Brief in the spec's ten-section format."""
from __future__ import annotations

from gios.core.schemas import Hypothesis, Learning, Opportunity, Signal
from gios.modules.growth_council_insight_brief.pipeline import Brief
from gios.modules.qualified_demand_leakage_auditor.analysis import OWNERS

SIGNAL_LABELS = [("customer", "Customer"), ("behavioral", "Behavioral"), ("funnel", "Funnel"),
                 ("revenue", "Revenue"), ("sales_operational", "Sales / operational")]


def _evidence(ids, by_id, limit: int = 220) -> list[str]:
    out = []
    for i in ids:
        x = by_id.get(i)
        if isinstance(x, Signal):
            text = x.summary if len(x.summary) <= limit else x.summary[: limit - 1] + "…"
            out.append(f"  - `{i}` ({x.evidence_status}): {text}")
        elif x is not None:
            out.append(f"  - `{i}`")
    return out


def to_markdown(b: Brief) -> str:
    d, by_id = b.draft, b.inputs.by_id()
    c = b.inputs.counts
    lines = [f"# Growth Council Insight Brief: {b.period}", "",
             f"_From the GIOS store: {c['signals']} signals, {c['hypotheses']} hypotheses, {c['roadmap']} roadmap items, "
             f"{c['experiments']} experiments and {c['learnings']} learnings. Signal → Why it matters → Evidence → "
             "Hypothesis → Recommended action → Owner → Learning. Numbers come from the cited records._", ""]

    lines += ["### 1. What changed?", d.what_changed, ""]
    lines += ["### 2. Why does it matter?", d.why_it_matters, ""]
    lines += ["### 3. What signals support it?"]
    for key, label in SIGNAL_LABELS:
        line = getattr(d.signals, key)
        lines += [f"- {label}: {line.text}"] + _evidence(line.ids, by_id)
    lines += [""]

    h = by_id.get(d.hypothesis_id or "")
    score = ""
    if isinstance(h, Hypothesis) and h.validation_total is not None:
        score = f" Validator score {h.validation_total}/12 ({(h.validator_recommendation or '').replace('_', ' ')})."
    hyp = d.working_hypothesis.rstrip(".") + (f" (`{d.hypothesis_id}`)." if d.hypothesis_id else ".")
    lines += ["### 4. Working hypothesis", hyp + score, ""]

    o = by_id.get(d.opportunity_id or "")
    roadmap = ""
    if isinstance(o, Opportunity):
        roadmap = (f" Roadmap: **{o.title}**, {o.horizon.title() if o.horizon else 'unscheduled'}, priority score "
                   f"{o.priority_score:,.0f}.")
    lines += ["### 5. Recommended action", d.recommended_action + roadmap, ""]
    deps = f" Flagged dependencies: {', '.join(o.dependencies)}." if isinstance(o, Opportunity) and o.dependencies else ""
    lines += ["### 6. Cross-functional dependency", d.cross_functional_dependency + deps, ""]

    suggested = ", ".join(OWNERS[s] for s in b.suggestions) or "none"
    lines += ["### 7. Owner", f"**{b.owner_label}**. {d.owner_rationale}",
              f"_Suggested from the Leakage Auditor owner classes: {suggested}._", ""]

    metric = f" Roadmap primary metric: {o.primary_metric}" if isinstance(o, Opportunity) and o.primary_metric else ""
    lines += ["### 8. Measurement", d.measurement + metric, ""]

    lessons = [by_id[i] for i in d.learning_ids if isinstance(by_id.get(i), Learning)]
    lines += ["### 9. What should the organization learn?", d.organization_learning]
    lines += [f"- From **{x.experiment_name}** ({x.decision.replace('_', ' ')}): {x.reusable_principle}"
              for x in lessons]
    lines += [""]
    lines += ["### 10. Decision needed from Growth Council", d.decision_needed]
    return "\n".join(lines)
