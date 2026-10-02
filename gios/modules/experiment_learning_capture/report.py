"""Render a learning in the spec's template (experiment-learning-capture.md)."""
from __future__ import annotations

from gios.core.schemas import Learning
from gios.modules.downstream_impact_analyzer.analysis import INTERPRETATION_LABELS

DECISION_LABELS = {"scale": "Scale", "iterate": "Iterate", "retest": "Retest", "stop": "Stop",
                   "research": "Research", "observe_longer": "Observe longer"}
NEXT_PARTS = [("observed_problem", "Observed problem"), ("affected_audience", "Affected audience"),
              ("causal_explanation", "Cause"), ("intervention", "Intervention"),
              ("expected_behavior_change", "Expected behavior change"),
              ("expected_business_outcome", "Expected business outcome")]


def to_markdown(x: Learning) -> str:
    interp = INTERPRETATION_LABELS.get(x.interpretation or "", "")
    facets = " · ".join(v for v in [x.theme and f"theme: {x.theme}", x.page and f"page: {x.page}",
                                    x.segment and f"segment: {x.segment}"] if v)
    lines = [f"# Experiment Learning: {x.experiment_name or x.experiment_id}", "", f"_{facets}_" if facets else "", ""]
    sections = [
        ("Experiment", f"{x.experiment_name} (`{x.experiment_id}`)"),
        ("Original Problem", x.original_problem),
        ("Hypothesis", x.hypothesis_text),
        ("What Happened", x.what_happened),
        ("Statistical / Directional Result", x.statistical_result or (f"**{interp}.**" if interp else "")),
        ("Segment Findings", x.segment_findings),
        ("What We Learned About the Customer", x.learned_about_customer),
        ("What We Learned About the Journey", x.learned_about_journey),
        ("What We Learned About the Business", x.learned_about_business),
        ("What We Should NOT Conclude", x.should_not_conclude),
        ("Decision", f"**{DECISION_LABELS[x.decision]}**"),
        ("Reusable Principle", x.reusable_principle),
    ]
    for heading, body in sections:
        lines += [f"### {heading}", body or "_Not captured._", ""]
    lines += ["### Next Hypothesis"]
    p = x.next_hypothesis_parts
    if p:
        lines += [f"**{p.get('title', x.next_hypothesis)}**", ""] + [f"- {label}: {p[k]}" for k, label in NEXT_PARTS
                                                                      if p.get(k)]
    else:
        lines += [x.next_hypothesis or "_Not captured._"]
    if x.next_hypothesis_id:
        lines += ["", f"_Sent to the Hypothesis Evidence Validator as `{x.next_hypothesis_id}`._"]
    return "\n".join(lines)
