"""Render the Growth Intelligence Diagnostic in the spec's output format."""
from __future__ import annotations

from gios.core.report import md_bullets
from gios.core.schemas import Signal
from gios.modules.growth_intelligence_diagnostic.analysis import EVIDENCE_GROUPS
from gios.modules.growth_intelligence_diagnostic.models import ACTION_LABELS, EvidenceClaim
from gios.modules.growth_intelligence_diagnostic.pipeline import DiagnosisResult
from gios.modules.qualified_demand_leakage_auditor.analysis import OWNERS

BADGE_UI = {"observed": ":green-badge[Observed]", "inferred": ":orange-badge[Inferred]",
            "unknown": ":gray-badge[Unknown]"}
BADGE_MD = {"observed": "`[Observed]`", "inferred": "`[Inferred]`", "unknown": "`[Unknown]`"}


def _claim(c: EvidenceClaim, badges: dict) -> str:
    return f"{badges[c.status]} {c.claim} (`{c.signal_id}`)"


def _signal(s: Signal, badges: dict) -> str:
    return f"{badges[s.evidence_status]} `{s.type}` `{s.id}`: {s.summary}"


def to_markdown(result: DiagnosisResult, ui: bool = False) -> str:
    badges = BADGE_UI if ui else BADGE_MD
    b, f, o = result.business, result.filters, result.output
    scope = ", ".join(x for x in [f.channel and f"channel {f.channel}", f.segment and f"segment {f.segment}",
                                  f"{f.start} to {f.end}"] if x)
    lines = ["# Growth Intelligence Diagnostic: EchoAI", "",
             f"_Scope: {scope}. {len(result.signals)} signals in scope. Leakage point from the Qualified Demand "
             "Leakage Auditor; hypotheses ranked in code by supporting minus contradicting signal strength._", ""]

    lines += ["### Growth Problem", o.growth_problem, "",
              md_bullets([f"**Goal:** {b.goal}", f"**Target metric:** {b.target_metric}",
                          f"**Underperforming:** {b.what}", f"**Baseline:** {b.baseline}",
                          f"**Audience / segment:** {b.segment}", f"**Period:** {b.period}"]), ""]

    cited = list(dict.fromkeys(c.signal_id for r in result.ranked
                               for c in r.root_cause.supporting + r.root_cause.contradicting))
    by_id = result.by_id
    lines += ["### Evidence"]
    for heading, types in EVIDENCE_GROUPS.items():
        items = [_signal(by_id[i], badges) for i in cited if by_id[i].type in types]
        lines += ["", f"**{heading}:**  ", md_bullets(items, empty="_No cited signals of this type._")]
    uncited = len(result.signals) - len(cited)
    if uncited:
        lines += ["", f"_{uncited} other signals in scope were not cited by any hypothesis._"]
    lines += [""]

    lines += ["### Likely Root Causes"]
    for r in result.ranked:
        h = r.root_cause
        lines += [f"{r.rank}. **{h.title}**: confidence **{r.confidence.title()}** (evidence score {r.score}; "
                  f"signal types: {', '.join(r.supporting_types)})",
                  f"   - Suspected cause: {h.causal_explanation}",
                  "   - Supporting evidence:"] + [f"     - {_claim(c, badges)}" for c in h.supporting]
        lines += ["   - Contradictory evidence:"] + ([f"     - {_claim(c, badges)}" for c in h.contradicting]
                                                     or ["     - _None found._"])
        lines += ["   - Missing evidence:"] + ([f"     - {badges['unknown']} {m}" for m in h.missing]
                                               or ["     - _None listed._"])
    lines += [""]

    p, top = result.leakage_point, result.top.root_cause
    lines += ["### Highest-Priority Opportunity"]
    if p:
        lines += [f"**Leakage point (deterministic): {p.source} · {p.stage}.** {p.describe()}", ""]
    else:
        lines += ["_No stage in this scope is significantly below benchmark._", ""]
    lines += [f"**Top hypothesis:** {top.title}. {top.observed_problem} Intervention: {top.intervention}", ""]

    lines += ["### Recommended Action", f"**{ACTION_LABELS[o.next_best_action]}**: {o.action_detail}", ""]
    lines += ["### Measurement Plan", md_bullets([f"Primary: {o.primary_metric}", f"Guardrail: {o.guardrail_metric}",
                                                  f"Downstream: {o.downstream_metric}",
                                                  f"Learning objective: {o.learning_objective}"]), ""]
    lines += ["### Confidence", f"**{result.top.confidence.title()}**", "",
              "_Computed in code for the top hypothesis: High needs at least three supporting signal types, two of "
              "them observed; Medium needs two types; anything less is Low. Contradictory evidence lowers it one "
              "level._", ""]
    missing = list(dict.fromkeys(list(o.missing_evidence) + [m for r in result.ranked for m in r.root_cause.missing]))
    lines += ["### Missing Evidence", md_bullets(f"{badges['unknown']} {m}" for m in missing)]
    return "\n".join(lines)


__all__ = ["to_markdown", "OWNERS"]
