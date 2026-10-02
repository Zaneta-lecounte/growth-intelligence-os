"""Render the Growth Priority Orchestrator in the spec's output format."""
from __future__ import annotations

from gios.core.report import md_bullets, md_table
from gios.core.schemas import Opportunity
from gios.modules.experiment_opportunity_scorer.analysis import RECOMMENDATION_LABELS
from gios.modules.growth_priority_orchestrator.analysis import BUCKETS, CATEGORY_LABELS, DEPENDENCY_LABELS
from gios.modules.growth_priority_orchestrator.pipeline import Roadmap


def _score(o: Opportunity) -> str:
    s = o.priority_score
    return "—" if s is None else f"{s:,.0f}" if s >= 10 else f"{s:,.2f}"


def _item(o: Opportunity) -> str:
    deps = ", ".join(DEPENDENCY_LABELS[d] for d in o.dependencies) or "none"
    merged = f" Merged with {len(o.merged_ids)} duplicate(s)." if o.merged_ids else ""
    return (f"**{o.title}** · {BUCKETS[o.portfolio_bucket]} · {CATEGORY_LABELS[o.category]} · score {_score(o)} · "
            f"{RECOMMENDATION_LABELS[o.final_recommendation]}. Dependencies: {deps}.{merged}")


def to_markdown(r: Roadmap) -> str:
    lines = ["# Growth Priority Orchestrator: EchoAI", "",
             f"_{len(r.active)} roadmap items after merging duplicates ({len(r.items) - len(r.active)} merged). "
             "Priority scores and recommendations come from the Experiment Opportunity Scorer; buckets, balance and "
             "default horizons are computed in code and can be edited._", ""]

    now = r.lane("now")
    lines += ["### Executive Growth Priorities"]
    top = now[:3] if now else sorted(r.active, key=lambda o: -(o.priority_score or 0))[:3]
    lines += [f"{i}. {_item(o)}" + (f" Primary metric: {o.primary_metric}" if o.primary_metric else "")
              for i, o in enumerate(top, 1)] or ["_No items on the roadmap._"]
    lines += [""]

    for horizon, label in (("now", "Now"), ("next", "Next"), ("later", "Later")):
        lines += [f"### {label}", md_bullets([_item(o) for o in r.lane(horizon)], empty="_Nothing scheduled._"), ""]

    research = []
    for o in r.active:
        h = r.hypotheses.get(o.hypothesis_id or "")
        if o.final_recommendation in ("research_first", "instrument_first"):
            needs = ("; ".join(m.rstrip(".") for m in h.missing_evidence[:2]) + "."
                     if h and h.missing_evidence else "see the validator report")
            research.append(f"**{o.title}** ({RECOMMENDATION_LABELS[o.final_recommendation]}): {needs}")
        elif "sample_size" in o.risk_flags and o.horizon:
            check = r.checks.get(o.id)
            weeks = f"~{check.weeks:.0f} weeks" if check and check.available else "too long"
            research.append(f"**{o.title}**: the sample takes {weeks} at {o.mde:.0%} MDE; use a larger expected "
                            "effect, a longer test, or a staged rollout with pre/post measurement.")
    lines += ["### Research / Instrumentation Required", md_bullets(research), ""]

    deps = []
    for d, label in DEPENDENCY_LABELS.items():
        users = [o.title for o in r.active if o.horizon and d in o.dependencies]
        if users:
            deps.append(f"**{label}**: " + "; ".join(users))
    lines += ["### Key Dependencies", md_bullets(deps), ""]

    learning = [f"**{o.title}**: {o.expected_learning}" for o in now + r.lane("next") if o.expected_learning]
    lines += ["### Expected Learning", md_bullets(learning), ""]

    metrics = [f"**{o.title}**: {o.primary_metric}" for o in r.active if o.horizon and o.primary_metric]
    for rec in r.recommendations:
        metrics.append(f"Diagnostic measurement plan ({rec.title}). Primary: {rec.primary_metric} Guardrail: "
                       f"{rec.guardrail_metric} Downstream: {rec.downstream_metric}")
    lines += ["### Metrics", md_bullets(metrics), ""]

    b = r.balance
    lines += ["---", "#### Appendix: portfolio balance", md_table(
        ["Bucket", "Items", "Share"], [[BUCKETS[k], b.counts[k], f"{b.shares[k]:.0%}"] for k in BUCKETS],
        align=["l", "r", "r"]), ""]
    lines += [md_bullets([f"⚠ {w}" for w in b.warnings] + b.notes, empty="_Balanced: no bucket above 50%._"), ""]
    if r.merges:
        titles = {o.id: o.title for o in r.items}
        lines += ["#### Appendix: merged duplicates", md_bullets(
            " + ".join(f"`{i}`" for i in ids) + f" → **{title or titles.get(ids[0], ids[0])}**" for ids, title in r.merges)]
    return "\n".join(lines)
