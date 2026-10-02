"""Render the Downstream Impact Analyzer in the spec's output format. All text is computed."""
from __future__ import annotations

import math

from gios.core.report import md_bullets, md_table
from gios.modules.downstream_impact_analyzer.analysis import (
    INTERPRETATION_LABELS,
    RECOMMENDATION_LABELS,
    RULES,
    ImpactResult,
    StageResult,
    recommend,
)


def _row(s: StageResult) -> list:
    t = s.test
    return [s.label, f"{t['p_control']:.2%}", f"{t['p_treatment']:.2%}", f"{t['lift']:+.1%}",
            f"[{t['ci_low']:+.2%}, {t['ci_high']:+.2%}]", f"{t['p_value']:.3f}",
            "yes" if t["significant"] else "no", f"{s.mde:.0%}" if math.isfinite(s.mde) else "∞",
            f"⚠ {s.events_per_arm}" if s.underpowered else str(s.events_per_arm)]


HEAD = ["Stage", "Control", "Treatment", "Lift", "Diff 95% CI", "p", "Sig.", "MDE (80%)", "Events/arm"]


def statistical_summary(r: ImpactResult) -> str:
    p = r.stage("conversions")
    sql = r.stage("sqls")
    return (f"{p.describe()}. {sql.describe()}. Interpretation: {INTERPRETATION_LABELS[r.interpretation]}.")


def to_markdown(r: ImpactResult) -> str:
    m = r.money
    lines = [f"# Downstream Impact Analyzer: {r.name}", "",
             f"_`{r.experiment_id}` · {r.control} vs {r.treatment} · two-proportion tests per visitor at every "
             f"stage, α = {r.settings.alpha:g}; stages with fewer than {r.settings.min_events} events per arm are "
             "flagged underpowered. Interpretation and recommendation follow documented rules, not the LLM._", ""]

    lines += ["### Immediate Result", md_table(HEAD, [_row(r.stage("conversions"))], align=["l"] + ["r"] * 8), ""]

    lines += ["### Downstream Result", md_table(HEAD, [_row(s) for s in r.stages[1:]], align=["l"] + ["r"] * 8), ""]
    cac = lambda v: "∞ (no wins)" if math.isinf(v) else f"${v:,.0f}"  # noqa: E731
    cac_change = "" if math.isnan(m.cac_change) else f" ({m.cac_change:+.0%})"
    lines += [md_bullets([
        f"Revenue per visitor: ${m.rpv_control:,.2f} → ${m.rpv_treatment:,.2f} (diff 95% CI "
        f"[${m.rpv_ci[0]:+,.2f}, ${m.rpv_ci[1]:+,.2f}], assuming equal deal size across arms)",
        f"CAC (traffic spend / wins): {cac(m.cac_control)} → {cac(m.cac_treatment)}{cac_change}",
    ])]
    under = [s.label for s in r.stages if s.underpowered]
    if under:
        lines += ["", f"> ⚠ Underpowered: {', '.join(under)} had fewer than {r.settings.min_events} events per arm. "
                      "Treat those stages as directional only."]
    if r.window_days is not None and r.velocity_days is not None:
        verdict = "shorter than" if r.needs_longer else "at least"
        lines += ["", f"Observation window: {r.window_days} days, {verdict} the traffic's median lead→win velocity "
                      f"of {r.velocity_days:.0f} days."]
    lines += [""]

    lines += ["### Quality Tradeoff", md_table(HEAD, [_row(q) for q in r.quality], align=["l"] + ["r"] * 8), ""]
    worse = [q.label for q in r.quality if q.test["significant"] and q.test["diff"] < 0]
    lines += [f"Quality rates significantly worse in {r.treatment}: {', '.join(worse)}." if worse
              else "No quality rate between stages is significantly worse.", ""]

    lines += ["### Segment Differences"]
    if r.segments:
        lines += [md_bullets(f"⚠ {n}" for n in r.segment_notes) if r.segment_notes
                  else "No segment differs from the overall result.", "",
                  md_table(["Segment", "Traffic", "Primary lift", "Primary sig.", "SQL/visitor lift", "SQL sig.",
                            "Differs from rest"],
                           [[f"{f.dimension} = {f.value}", f"{f.share:.0%}", f"{f.primary['lift']:+.0%}",
                             "yes" if f.primary["significant"] else "no", f"{f.sql['lift']:+.0%}",
                             "yes" if f.sql["significant"] else "no",
                             f"⚠ z = {f.interaction_z:+.1f}" if f.differs else "no"] for f in r.segments],
                           align=["l", "r", "r", "l", "r", "l", "l"])]
    else:
        lines += ["_No segment breakdown available for this experiment._"]
    lines += [""]

    lines += ["### Business Interpretation", f"**{INTERPRETATION_LABELS[r.interpretation]}**. Rule: {r.rule}", ""]
    _, why = recommend(r.interpretation, r.stage("conversions"), r.settings.meaningful_lift)
    lines += ["### Recommendation", f"**{RECOMMENDATION_LABELS[r.recommendation]}**. {why}", "",
              "#### Interpretation rules (applied in order)", md_bullets(RULES)]
    return "\n".join(lines)
