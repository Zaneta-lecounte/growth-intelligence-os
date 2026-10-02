"""Deterministic parts of the Growth Council Insight Brief: gathering the period's inputs from
the store, owner suggestions from the Leakage Auditor's owner classes, and the Slack summary."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from gios.core.schemas import Experiment, Hypothesis, Learning, Opportunity, Recommendation, Signal
from gios.core.store import Store
from gios.modules.qualified_demand_leakage_auditor.analysis import OWNERS

MODULE = "growth_council_insight_brief"
SLACK_MAX_LINES = 8
CATEGORY_OWNER = {"conversion": "web_conversion", "acquisition": "acquisition", "qualification": "qualification",
                  "operations_measurement": "operations_routing", "revenue": "sales_handoff",
                  "activation": "product_offering", "retention": "product_offering",
                  "customer_problem": "product_offering"}


@dataclass
class BriefInputs:
    start: str
    end: str
    signals: list[Signal]
    hypotheses: list[Hypothesis]
    roadmap: list[Opportunity]
    recommendations: list[Recommendation]
    experiments: list[Experiment]
    learnings: list[Learning]
    counts: dict[str, int] = field(default_factory=dict)

    def allowed_ids(self) -> frozenset[str]:
        return frozenset(x.id for group in (self.signals, self.hypotheses, self.roadmap, self.experiments,
                                            self.learnings) for x in group)

    def by_id(self) -> dict:
        return {x.id: x for group in (self.signals, self.hypotheses, self.roadmap, self.experiments, self.learnings)
                for x in group}


def _month(d) -> str:
    return d.strftime("%Y-%m") if d else ""


def gather(store: Store, start: str, end: str) -> BriefInputs:
    """Signals whose period overlaps [start, end]; experiments that ended in the period and their
    learnings; plus the current diagnostics, validator scores and roadmap."""
    signals = [s for s in store.list(Signal) if s.covers_month_range(start, end)]
    experiments = [e for e in store.list(Experiment) if e.end_date and start <= _month(e.end_date) <= end]
    exp_ids = {e.id for e in experiments}
    learnings = [x for x in store.list(Learning) if x.experiment_id in exp_ids]
    roadmap = sorted([o for o in store.list(Opportunity) if o.horizon and not o.merged_into],
                     key=lambda o: ("now", "next", "later").index(o.horizon) * 100 + (o.horizon_order or 0))
    inputs = BriefInputs(start, end, signals, store.list(Hypothesis), roadmap, store.list(Recommendation),
                         experiments, learnings)
    inputs.counts = {"signals": len(signals), "hypotheses": len(inputs.hypotheses), "roadmap": len(roadmap),
                     "experiments": len(experiments), "learnings": len(learnings)}
    return inputs


def owner_suggestions(cited: Iterable[str], inputs: BriefInputs) -> list[str]:
    """Owner classes from the Leakage Auditor: owners named on cited leakage signals first, then the
    owner implied by the cited roadmap item's category; most frequent first."""
    by_id = inputs.by_id()
    counts: Counter = Counter()
    for i in cited:
        item = by_id.get(i)
        if isinstance(item, Signal) and item.attributes.get("owner") in OWNERS:
            counts[item.attributes["owner"]] += 2
        elif isinstance(item, Opportunity):
            counts[CATEGORY_OWNER.get(item.category, "product_offering")] += 3
    return [o for o, _ in counts.most_common()]


def first_sentence(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    m = re.match(r"(.+?[.!?])(\s|$)", text)
    s = m.group(1) if m else text
    return s if len(s) <= limit else s[: limit - 1].rstrip() + "…"


def slack_summary(period: str, what_changed: str, why: str, hypothesis: str, action: str, owner: str,
                  measurement: str, decision: str) -> str:
    """A Slack-ready summary of at most eight lines."""
    lines = [
        f":bar_chart: *Growth Council brief* · {period}",
        f"*What changed:* {first_sentence(what_changed)}",
        f"*Why it matters:* {first_sentence(why)}",
        f"*Hypothesis:* {first_sentence(hypothesis)}",
        f"*Action:* {first_sentence(action)} _(owner: {owner})_",
        f"*Measure:* {first_sentence(measurement)}",
        f"*Decision needed:* {first_sentence(decision)}",
        "Full brief attached (Markdown).",
    ]
    if len(lines) > SLACK_MAX_LINES:  # guard the contract even under python -O
        raise ValueError(f"Slack summary must be at most {SLACK_MAX_LINES} lines")
    return "\n".join(lines)
