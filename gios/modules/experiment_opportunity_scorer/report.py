"""Render the Experiment Opportunity Scorer in the spec's output format."""
from __future__ import annotations

from gios.core.report import md_bullets, md_table
from gios.modules.experiment_opportunity_scorer.analysis import RECOMMENDATION_LABELS, RULES
from gios.modules.experiment_opportunity_scorer.pipeline import ScoredBacklog

JUDGMENT_NOTE = ("The score supports judgment; it does not replace it. Check strategic fit, measurement "
                 "readiness, and risk before acting, and override the recommendation with a reason when warranted.")


def _score(x) -> str:
    return "—" if x is None else (f"{x:,.0f}" if x >= 10 else f"{x:,.2f}")


def to_markdown(sb: ScoredBacklog) -> str:
    ranked = sb.ranked()
    lines = ["# Experiment Opportunity Scorer: EchoAI", "",
             "_Priority Score = (G × R × O × W × H) / T, computed in code (range 0.2 to 3,125). G and T are "
             "judgment inputs; R, O, W and H are prefilled from linked evidence and the validator. All are editable._",
             "", f"> {JUDGMENT_NOTE}", ""]
    lines += [md_table(
        ["Opportunity", "G", "R", "O", "W", "T", "H", "Score", "Key Risk"],
        [[o.title, o.growth_impact, o.research_evidence, o.opportunity_size, o.web_evidence, o.technical_effort,
          o.hypothesis_confidence, _score(o.priority_score), o.key_risk] for o in ranked],
        align=["l"] + ["r"] * 7 + ["l"]), ""]

    lines += ["### Recommendation"]
    for key, label in RECOMMENDATION_LABELS.items():
        names = [o.title + (" _(override)_" if o.recommendation_override else "")
                 for o in ranked if o.final_recommendation == key]
        lines += [f"- **{label}**: " + ("; ".join(names) if names else "—")]
    lines += [""]

    lines += ["### Rationale"]
    for i, o in enumerate(ranked, 1):
        check = sb.checks.get(o.id)
        bits = [f"Rule: {sb.rules.get(o.id, '')}"]
        if o.validator_total is not None:
            bits.append(f"Validator total {o.validator_total}/12.")
        if o.recommendation_override:
            bits.append(f"**Overridden** from {RECOMMENDATION_LABELS[o.recommendation]} to "
                        f"{RECOMMENDATION_LABELS[o.recommendation_override]}: {o.override_reason}")
        if check:
            bits.append(f"Sample size: {check.describe(sb.settings.max_weeks)}.")
        lines += [f"{i}. **{o.title}** → {RECOMMENDATION_LABELS[o.final_recommendation]} "
                  f"(score {_score(o.priority_score)}). " + " ".join(bits)]
    lines += ["", "#### Recommendation rules (applied in order)", md_bullets(RULES)]
    return "\n".join(lines)
