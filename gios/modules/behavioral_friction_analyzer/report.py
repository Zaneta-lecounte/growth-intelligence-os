"""Render Behavioral Friction Analyzer output in the spec's format, one block per finding."""
from __future__ import annotations

from gios.core.report import md_bullets, money, num, pct
from gios.modules.behavioral_friction_analyzer.models import ACTION_LABELS, FRICTION_LABELS
from gios.modules.behavioral_friction_analyzer.pipeline import BehavioralResult

NOT_CLASSIFIED = ("_Not classified: no LLM output for this finding. In demo mode only findings at "
                  "the default thresholds have cached classifications._")


def _impact(sizing: dict) -> str:
    if not sizing.get("sized"):
        return ("Not sized: the flagged metrics (rage clicks, load time, or trends) have no direct "
                "conversion path. Treat as a diagnostic that supports other findings.")
    return (
        f"~{num(sizing['lost_leads_per_month'])} lost leads/month → ~{sizing['lost_wins_per_month']:.1f} wins → "
        f"~{money(sizing['lost_revenue_per_month'])}/month in revenue at current downstream rates "
        f"(lead→win {pct(sizing['lead_to_win'])}, revenue/win {money(sizing['revenue_per_win'])}).\n\n"
        f"Basis: {sizing['basis']} (~{num(sizing['lost_events_per_month'])}/month). Losses across metrics "
        "on the same slice overlap, so only the largest is counted. This is an upper bound: it assumes "
        "no visitor recovers through another path or device."
    )


def to_markdown(result: BehavioralResult) -> str:
    th = result.thresholds
    lines = ["# Behavioral Friction Analyzer: EchoAI", "",
             f"_{num(result.meta.get('sessions', 0))} sessions, {result.meta.get('months', '')}. Flags need "
             f"|z| ≥ {th.z:g}, deviation ≥ {th.min_deviation:.0%}, volume ≥ {num(th.min_volume)} "
             f"(load time: ≥ {th.load_deviation:.0%} slower than the same device's other pages). Baselines are "
             "standardized for device and source mix._", ""]
    if result.meta.get("llm_error"):
        lines += [f"> Friction classification unavailable: {result.meta['llm_error']}", ""]
    if not result.findings:
        return "\n".join(lines + ["_No behavioral anomalies cleared the thresholds._"])
    lines += [f"{i}. {f.title}" for i, f in enumerate(result.findings, 1)] + [""]

    for i, f in enumerate(result.findings, 1):
        c = result.classifications.get(f.id)
        ev = result.evidence.get(f.id)
        lines += ["---", f"## Finding {i}: {f.title}", ""]

        observed = [fl.describe() for fl in f.flags]
        if "form_start_rate" in f.context:
            observed.append(f"Form start rate {pct(f.context['form_start_rate'])} vs "
                            f"{pct(f.context['form_start_rate_rest'])} elsewhere on the page")
        lines += ["### Behavioral Signal", md_bullets(observed)]
        if c:
            lines += ["", c.what_happened]
        lines += [""]

        lines += ["### Friction Type"]
        if c:
            linked = ""
            if ev and ev.linked_signal_ids:
                linked = " (" + ", ".join(f"`{s}`" for s in ev.linked_signal_ids) + ")"
            lines += [f"**{FRICTION_LABELS[c.friction_type]}**. Cause status: {ev.status}{linked}."]
        else:
            lines += [NOT_CLASSIFIED]
        lines += [""]

        mix = sorted(f.context.get("source_mix", {}).items(), key=lambda kv: -kv[1])[:3]
        mix_text = ", ".join(f"{s} {w:.0%}" for s, w in mix)
        lines += ["### Affected Segment",
                  f"{f.page.title()} page · {f.dimension} = {f.value} · ~{num(f.context.get('sessions_per_month', 0))} "
                  f"sessions/month ({pct(f.context.get('share_of_page_sessions', 0), 0)} of page sessions) · "
                  f"top sources: {mix_text}", ""]

        lines += ["### Likely Business Impact", _impact(f.sizing), ""]
        lines += ["### Competing Explanations",
                  md_bullets(c.competing_explanations) if c else NOT_CLASSIFIED, ""]
        validation = list(c.validation_needed) if c else []
        if ev and not ev.linked_signal_ids:
            validation.insert(0, "No customer evidence is linked yet, so no cause is claimed. Run the "
                                 "Customer Signal Synthesizer and save its signals to connect evidence.")
        if ev and ev.dropped_signal_ids:
            validation.append("Ignored citations to unknown customer signals: "
                              + ", ".join(ev.dropped_signal_ids))
        lines += ["### Validation Needed", md_bullets(validation) if c else NOT_CLASSIFIED, ""]
        lines += ["### Recommended Experiment / Research",
                  f"**{ACTION_LABELS[c.recommended_action]}**: {c.recommended_detail}" if c else NOT_CLASSIFIED, ""]
    return "\n".join(lines)
