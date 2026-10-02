"""Render the Hypothesis Evidence Validator in the spec's output format."""
from __future__ import annotations

from gios.core.report import md_bullets, md_table
from gios.modules.hypothesis_evidence_validator.analysis import BAND_LABELS, RECOMMENDATION_LABELS, SIGNAL_TYPES
from gios.modules.hypothesis_evidence_validator.models import DIMENSION_LABELS, RUBRIC
from gios.modules.hypothesis_evidence_validator.pipeline import ValidationResult

STATUS_MARK = {"ok": "✓", "missing": "✗ missing", "vague": "⚠ vague"}


def _statement(r: ValidationResult) -> str:
    parts = {c.part: c.text or f"[{c.label.lower()} missing]" for c in r.standard}
    return (f"If we **{parts['intervention']}** for **{parts['affected_audience']}**, then "
            f"**{parts['expected_behavior_change']}**, because **{parts['causal_explanation']}**, leading to "
            f"**{parts['expected_business_outcome']}**. Observed problem: {parts['observed_problem']}.")


def to_markdown(r: ValidationResult) -> str:
    h = r.hypothesis
    lines = ["# Hypothesis Evidence Validator: EchoAI", "",
             f"_Hypothesis `{h.id}`{' (' + h.title + ')' if h.title else ''}. The LLM proposes each 0–2 score; "
             "code caps it to what the linked evidence supports; the user may override. Total, band and "
             "recommendation are computed in code._", ""]

    lines += ["### Hypothesis", _statement(r), "", md_table(
        ["Part", "Status", "Text"], [[c.label, STATUS_MARK[c.status], c.text or "—"] for c in r.standard]), ""]

    em = r.evidence
    rows = []
    for t in SIGNAL_TYPES:
        sup = ", ".join(f"`{s.id}` ({s.evidence_status})" for s in em.supporting[t]) or "—"
        con = ", ".join(f"`{s.id}` ({s.evidence_status})" for s in em.contradicting[t]) or "—"
        rows.append([t, "✓" if em.supporting[t] else "", sup, con])
    lines += ["### Evidence Map", md_table(["Signal type", "Supports", "Supporting signals", "Contradicting signals"],
                                           rows), ""]
    if em.unknown_ids:
        lines += [f"_Linked ids not found in the store (ignored): {', '.join(em.unknown_ids)}._", ""]

    def note(row):
        bits = []
        if row.capped and not row.overridden:
            bits.append(f"capped from {row.proposed}: {row.cap_reason}")
        if row.overridden:
            bits.append("user override" + (f"; above evidence cap ({row.cap_reason})" if row.above_cap else ""))
        return "; ".join(bits)

    lines += ["### Score", md_table(
        ["Dimension", "Score", "Justification", "Adjustment"],
        [[DIMENSION_LABELS[x.dimension], f"{x.score} ({RUBRIC[x.dimension][x.score]})", x.justification, note(x) or "—"]
         for x in r.rows], align=["l", "r", "l", "l"]), "",
              f"**Total: {r.total} / 12** ({BAND_LABELS[r.band]})", ""]

    w = r.weakest
    lines += ["### Weakest Evidence Area",
              f"**{DIMENSION_LABELS[w.dimension]}** ({w.score}/2: {RUBRIC[w.dimension][w.score]}). {w.justification}"
              + (f" Evidence cap: {w.cap_reason}." if w.cap_reason else ""), ""]
    lines += ["### Missing Evidence", md_bullets(r.missing_evidence), ""]
    lines += ["### Recommendation", f"**{RECOMMENDATION_LABELS[r.recommendation]}**", "",
              "_Rule: 9–12 Test; 5–8 Instrument first when measurement readiness is no stronger than the weakest "
              "evidence dimension, otherwise Research first; 0–4 Reject (not test-ready until evidence is linked)._"]
    return "\n".join(lines)
