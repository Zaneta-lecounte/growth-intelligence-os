"""Experiment Learning Capture: prefill from the stored experiment and its Downstream Impact result
(code) -> LLM drafts the narrative sections -> user edits -> save to the Experiment Library."""
from __future__ import annotations

import json
from typing import Any, Optional

from gios.core import data
from gios.core.llm import complete_json
from gios.core.schemas import Experiment, Hypothesis, Learning
from gios.core.store import Store
from gios.modules.downstream_impact_analyzer.analysis import INTERPRETATION_LABELS
from gios.modules.experiment_learning_capture.models import LearningDraft

MODULE = "experiment_learning_capture"
PROMPT = "draft_experiment_learning"
NARRATIVE_FIELDS = ["original_problem", "what_happened", "learned_about_customer", "learned_about_journey",
                    "learned_about_business", "should_not_conclude", "reusable_principle"]
DECISION_FROM_IMPACT = {"scale": "scale", "iterate": "iterate", "retest": "retest", "stop": "stop",
                        "observe_longer": "observe_longer"}
MIN_NOT_CONCLUDE = 10


def learning_id(experiment_id: str) -> str:
    return f"elc-{experiment_id}"


def dominant_segment(experiment_id: str) -> str:
    cells = data.load("experiment_segments")
    cells = cells[cells.experiment_id == experiment_id]
    if cells.empty:
        return "all"
    shares = cells.groupby("segment").visitors.sum() / cells.visitors.sum()
    return shares.idxmax() if shares.max() >= 0.65 else "all"


def prefill(e: Experiment) -> dict:
    """Deterministic fields, written by code from the experiment and its impact result."""
    return {
        "experiment_id": e.id, "experiment_name": e.name, "hypothesis_text": e.hypothesis_text,
        "statistical_result": e.statistical_result,
        "segment_findings": " ".join(f"{s}." for s in e.segment_findings) if e.segment_findings
        else "No segment differed significantly from the overall result.",
        "interpretation": e.interpretation, "page": e.page, "segment": dominant_segment(e.id),
        "decision": DECISION_FROM_IMPACT.get(e.impact_recommendation or "", "research"),
    }


def payload(e: Experiment) -> dict[str, str]:
    variants = [v.model_dump() for v in e.variants]
    return {
        "experiment": json.dumps({"name": e.name, "page": e.page, "hypothesis": e.hypothesis_text,
                                  "primary_metric": e.primary_metric,
                                  "start": str(e.start_date), "end": str(e.end_date),
                                  "interpretation": INTERPRETATION_LABELS.get(e.interpretation or "", "unknown"),
                                  "recommendation": e.impact_recommendation}, indent=2),
        "statistical_result": e.statistical_result,
        "segment_findings": "\n".join(e.segment_findings) or "None differed significantly.",
        "variants": json.dumps(variants, indent=2),
    }


def draft(e: Experiment, client: Any = None) -> LearningDraft:
    return complete_json(PROMPT, LearningDraft, payload(e), demo_key=e.id, client=client)


def build(e: Experiment, d: Optional[LearningDraft], edits: Optional[dict] = None) -> Learning:
    """Combine deterministic prefill, the LLM draft, and user edits (edits win)."""
    values = prefill(e)
    if d is not None:
        values |= {f: getattr(d, f) for f in NARRATIVE_FIELDS} | {"theme": d.theme}
        nh = d.next_hypothesis
        values["next_hypothesis_parts"] = nh.model_dump()
        values["next_hypothesis"] = f"{nh.title}: {nh.intervention}"
    values |= {k: v for k, v in (edits or {}).items() if v is not None}
    values.setdefault("what_happened", "")
    return Learning(id=learning_id(e.id), **values)


def check(learning: Learning) -> list[str]:
    """Problems that block saving. 'What We Should NOT Conclude' is required."""
    problems = []
    if len(learning.should_not_conclude.strip()) < MIN_NOT_CONCLUDE:
        problems.append("'What We Should NOT Conclude' is required.")
    if not learning.what_happened.strip():
        problems.append("'What Happened' is required.")
    return problems


def save(learning: Learning, store: Store) -> str:
    problems = check(learning)
    if problems:
        raise ValueError(" ".join(problems))
    return store.save(learning, module=MODULE)


def send_next_hypothesis(learning: Learning, store: Store) -> Hypothesis:
    """Create a Hypothesis record from the learning's next hypothesis, ready for the Validator."""
    parts = dict(learning.next_hypothesis_parts)
    if not parts:
        parts = {"title": learning.next_hypothesis, "observed_problem": learning.next_hypothesis}
    h = Hypothesis(id=f"elc-next-{learning.experiment_id}", **{k: v for k, v in parts.items()
                                                                 if k in Hypothesis.model_fields})
    store.save(h, module=MODULE)
    learning.next_hypothesis_id = h.id
    store.save(learning, module=MODULE)
    return h


def draft_all(store: Store, client: Any = None) -> list[str]:
    """Draft and save a learning for every analyzed experiment that has none yet."""
    ids = []
    for e in store.list(Experiment):
        if store.get(Learning, learning_id(e.id)) is None:
            ids.append(save(build(e, draft(e, client)), store))
    return ids
