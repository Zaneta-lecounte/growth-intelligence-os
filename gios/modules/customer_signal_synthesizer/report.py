"""Render Customer Signal Synthesizer output in the spec's format."""
from __future__ import annotations

import pandas as pd

from gios.core.report import md_bullets, md_table
from gios.modules.customer_signal_synthesizer.models import THEME_LABELS, ThemeSynthesis

TOP_N = 4


def research_gaps(themes: pd.DataFrame, synthesis: ThemeSynthesis, untagged: int,
                  disagreements: dict[str, int]) -> list[str]:
    gaps = list(synthesis.research_gaps)
    if untagged:
        gaps.append(f"{untagged} evidence rows could not be tagged; review them manually.")
    if disagreements.get("journey_stage"):
        gaps.append(f"{disagreements['journey_stage']} verbatims imply a different journey stage than "
                    "their source metadata; check how stages are recorded.")
    if disagreements.get("segment"):
        gaps.append(f"{disagreements['segment']} verbatims imply a different segment than their metadata.")
    unassessed = themes[~themes.assessed].label.tolist() if "assessed" in themes else []
    if unassessed:
        gaps.append("Scores were not proposed for: " + ", ".join(unassessed) + " (default 3 used).")
    thin = themes.head(TOP_N)
    thin = thin[thin["count"] < 10]
    for r in thin.itertuples():
        gaps.append(f"{r.label} ranks in the top {TOP_N} on only {r.count} verbatims; gather more evidence.")
    return gaps


def to_markdown(themes: pd.DataFrame, synthesis: ThemeSynthesis, untagged: int = 0,
                disagreements: dict[str, int] | None = None, meta: dict | None = None) -> str:
    meta = meta or {}
    by_theme = {a.theme: a for a in synthesis.assessments}
    top = themes.head(TOP_N)
    lines = ["# Customer Signal Synthesizer: EchoAI", ""]
    if meta:
        lines += [f"_{meta.get('evidence_rows', '?')} evidence items ({meta.get('unique_verbatims', '?')} "
                  f"distinct verbatims), {meta.get('months', '')}. Ranked by overall score "
                  "(Frequency × Severity × Commercial relevance × Journey relevance)._", ""]

    lines += ["### Top Customer Themes", md_table(
        ["Theme", "Evidence", "Segment", "Journey Stage", "Frequency", "Severity"],
        [[r.label, f"{r.count} verbatims ({r.share * 100:.0f}%), {r.trend}: \"{r.quote}\"",
          r.segment_text, r.journey_stage_text, r.frequency, r.severity] for r in themes.itertuples()],
        align=["l", "l", "l", "l", "r", "r"],
    ), ""]

    lines += ["### Customer Tensions", md_bullets(synthesis.tensions), ""]

    signals = []
    for r in top.itertuples():
        a = by_theme.get(r.theme)
        if a:
            signals.append(f"**{r.label}**: {a.expected_behavior} Funnel stage: {a.funnel_stage}. "
                           f"Validate with: {a.analytics_signal}")
    lines += ["### Behavioral Signals to Validate", md_bullets(signals), ""]

    opps = []
    for r in top.itertuples():
        a = by_theme.get(r.theme)
        if a:
            opps.append(f"**{r.label}** (overall {r.overall}): {a.testable_question}")
    lines += ["### Growth Opportunities", md_bullets(opps), ""]

    lines += ["### Research Gaps",
              md_bullets(research_gaps(themes, synthesis, untagged, disagreements or {})), ""]

    lines += ["---", "#### Appendix: prioritization detail", md_table(
        ["Rank", "Theme", "Count", "F", "S", "C", "J", "S×C", "Overall", "Freq. rank"],
        [[r.overall_rank, r.label, r.count, r.frequency, r.severity, r.commercial_relevance,
          r.journey_relevance, r.severity_x_commercial, r.overall, r.frequency_rank]
         for r in themes.itertuples()],
        align=["r", "l"] + ["r"] * 8,
    ), "", "_Frequency is counted from LLM tags and binned by share of evidence (<2%, <5%, <10%, "
       "<20%, ≥20%). Severity, commercial and journey relevance are LLM-proposed and editable._"]
    return "\n".join(lines)


__all__ = ["to_markdown", "research_gaps", "THEME_LABELS"]
