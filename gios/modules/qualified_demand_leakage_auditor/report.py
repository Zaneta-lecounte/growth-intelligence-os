"""Render Qualified Demand Leakage Auditor output in the spec's format."""
from __future__ import annotations

from gios.core.report import md_bullets, md_table, money, num, pct
from gios.modules.qualified_demand_leakage_auditor.analysis import OWNERS, STAGES
from gios.modules.qualified_demand_leakage_auditor.pipeline import LeakageResult

STAGE_NAMES = [name for _, _, name in STAGES] + ["Lead→Win"]


def _ci(r) -> str:
    return f"{pct(r.rate)} [{pct(r.ci_low)}–{pct(r.ci_high)}]"


def to_markdown(result: LeakageResult) -> str:
    a, n = result.analysis, result.narrative
    s = a.settings
    top = a.top_leak
    lines = ["# Qualified Demand Leakage Auditor: EchoAI", "",
             f"_Stage map Visit → Lead → MQL → SQL → Opportunity → Win across {a.months} months. Rates carry "
             f"Wilson 95% CIs; benchmarks are pooled from all other sources; stages need ≥ {s.min_volume} in the "
             f"denominator before they can be flagged. Sizing closes {s.gap_closure:.0%} of the gap to "
             "benchmark and carries the extra conversions through the source's own downstream rates._", ""]
    if result.narrative_note:
        lines += [f"> {result.narrative_note}", ""]

    lines += ["### Leakage Summary"]
    if n:
        lines += [n.leakage_summary, ""]
    lines += [md_bullets(f"**{f.label}** ({f.source}): {f.evidence}" for f in a.flags), ""]

    lines += ["### Highest-Value Leakage Point"]
    if top is None:
        lines += ["_No stage is significantly below benchmark with enough volume._", ""]
    else:
        lines += [
            f"**{top.source} · {top.stage}**: {pct(top.rate)} [{pct(top.ci_low)}–{pct(top.ci_high)}] on "
            f"{num(top.volume)} {top.stage.split('→')[0]}s vs a {pct(top.benchmark)} benchmark "
            f"({top.vs_benchmark:+.0%}).",
            "",
            f"Closing {s.gap_closure:.0%} of the gap (to {pct(top.lifted_rate)}) adds ~{num(top.extra_conversions_per_month)} "
            f"{top.stage.split('→')[1]}s → ~{top.extra_wins_per_month:.1f} wins → ~{money(top.extra_revenue_per_month)} "
            f"revenue per month (~{money(top.extra_revenue)} over the period).", ""]

    lines += ["### Evidence"]
    if top is not None:
        src = a.stage_map[a.stage_map.source == top.source].set_index("stage").loc[STAGE_NAMES]
        lines += [f"**{top.source} stage map vs other sources**", "", md_table(
            ["Stage", "Conversion [95% CI]", "Volume", "Benchmark", "Status"],
            [[st, _ci(r), num(r.denominator), pct(r.benchmark), r.status] for st, r in src.iterrows()],
            align=["l", "r", "r", "r", "l"]), ""]
        seg = a.segment_stage_map
        seg = seg[(seg.source == top.source) & (seg.stage == top.stage)]
        lines += [f"**{top.stage} by segment**", "", md_table(
            ["Segment", "Conversion [95% CI]", "Volume", "Segment benchmark", "Status"],
            [[r.segment, _ci(r), num(r.denominator), pct(r.benchmark), r.status] for r in seg.itertuples()],
            align=["l", "r", "r", "r", "l"]), ""]
        e = a.economics.loc[top.source]
        lines += [f"**{top.source} economics**: spend {money(e.spend)}, CPL {money(e.cpl)}, cost per SQL "
                  f"{money(e.cost_per_sql)}, CAC {money(e.cac)}, ROAS {e.roas:.1f}×.", ""]
        if top.source in a.rejections.index:
            r = a.rejections.loc[top.source]
            reasons = r.drop(["rejections", "qualification_share"]).astype(int).sort_values(ascending=False)
            lines += [f"**Sales rejection reasons ({top.source}, {num(r.rejections)} rejections)**: "
                      + ", ".join(f"{k} {v / r.rejections:.0%}" for k, v in reasons.items() if v) + "."]
        if top.source in a.follow_up.index:
            f = a.follow_up.loc[top.source]
            lines += [f"**Follow-up ({top.source})**: median {f.median_hours:.1f}h to first touch, "
                      f"{pct(f.late_share)} after the {s.sla_hours:g}h SLA."]
        lines += [""]

    lines += ["### Probable Cause Categories"]
    explained = {c.owner: c.explanation for c in n.probable_causes} if n else {}
    categories = list(dict.fromkeys(([top.owner] if top is not None else [])
                                    + [f.owner for f in a.flags if top is not None and f.source == top.source]
                                    + list(explained)))
    lines += [md_bullets(f"**{OWNERS[c]}**" + (f": {explained[c]}" if c in explained else "") for c in categories), ""]

    reasons: dict[str, list[str]] = {}
    if top is not None:
        reasons.setdefault(top.owner, []).append(f"{top.source} {top.stage} leak")
    for f in a.flags:
        reasons.setdefault(f.owner, []).append(f"{f.label} ({f.source})")
    lines += ["### Recommended Owner(s)", md_bullets(
        f"**{OWNERS[o]}**" + (" (primary)" if i == 0 else "") + ": " + "; ".join(dict.fromkeys(reasons.get(o, [])))
        for i, o in enumerate(a.owners())), ""]

    lines += ["### Recommended Intervention",
              n.recommended_intervention if n else "_Narrative unavailable; see the note above._", ""]

    lines += ["### Primary / Downstream Metrics"]
    if top is not None:
        src = a.stage_map[a.stage_map.source == top.source].set_index("stage")
        names = [name for _, _, name in STAGES]
        later = names[names.index(top.stage) + 1:]
        lines += [f"- **Primary**: {top.source} {top.stage}, now {_ci(top)}, target {pct(top.lifted_rate)}"
                  + (f". {n.primary_metric}" if n else "")]
        lines += [f"- **Downstream**: " + "; ".join(f"{st} {pct(src.loc[st].rate)}" for st in later + ["Lead→Win"])
                  + f"; cost per SQL {money(a.economics.loc[top.source].cost_per_sql)}; "
                    f"CAC {money(a.economics.loc[top.source].cac)}"]
        if n:
            lines += [f"- {m}" for m in n.downstream_metrics]
    lines += [""]

    lines += ["---", "#### Appendix: all sized leaks", md_table(
        ["Source", "Stage", "Conversion [95% CI]", "Benchmark", "Volume", "+Wins / month", "+Revenue / month", "Owner"],
        [[r.source, r.stage, _ci(r), pct(r.benchmark), num(r.volume), f"{r.extra_wins_per_month:.1f}",
          money(r.extra_revenue_per_month), OWNERS[r.owner]] for r in a.leaks.itertuples()],
        align=["l", "l", "r", "r", "r", "r", "r", "l"]), ""]
    pivot = a.stage_map.assign(cell=a.stage_map.apply(
        lambda r: _ci(r) + ("" if r.status in ("in line", "above") else " ▼" if r.status == "below" else " ?"), axis=1))
    pivot = pivot.pivot(index="source", columns="stage", values="cell")[STAGE_NAMES]
    lines += ["#### Appendix: stage map by source (▼ below benchmark, ? insufficient volume)", md_table(
        ["Source"] + STAGE_NAMES, [[src] + row.tolist() for src, row in pivot.iterrows()],
        align=["l"] + ["r"] * len(STAGE_NAMES))]
    return "\n".join(lines)
