"""Growth Priority Orchestrator pipeline: gather every module's stored output -> LLM proposes
duplicate clusters and qualitative tags -> user confirms merges and edits -> defaults, balance
and roadmap in code."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from gios.core.llm import LLMError, complete_json
from gios.core.schemas import Hypothesis, Opportunity, Recommendation, Signal
from gios.core.store import Store
from gios.modules.experiment_opportunity_scorer import Settings as ScorerSettings
from gios.modules.experiment_opportunity_scorer import pipeline as scorer
from gios.modules.growth_priority_orchestrator import analysis
from gios.modules.growth_priority_orchestrator.models import ItemTags, OrchestratorProposal

PROMPT = "propose_roadmap_clusters"


@dataclass
class Inputs:
    candidates: list[Opportunity]
    hypotheses: dict[str, Hypothesis]
    recommendations: list[Recommendation]
    signal_counts: dict[str, int]
    checks: dict[str, Any] = field(default_factory=dict)


def gather(store: Store, settings: ScorerSettings = ScorerSettings()) -> Inputs:
    """Opportunities from the scorer, plus any stored hypothesis not yet in the backlog (prefilled
    the same way), re-scored so recommendations follow the scorer's rules and overrides."""
    backlog = scorer.build_backlog(store)
    stored = {o.id: o for o in store.list(Opportunity)}
    for o in backlog:  # keep orchestrator fields saved on the record
        if o.id in stored:
            o.horizon, o.horizon_order = stored[o.id].horizon, stored[o.id].horizon_order
    scored = scorer.score(backlog, settings, store)
    signals = store.list(Signal)
    counts = {t: sum(s.type == t for s in signals) for t in
              ["customer", "behavioral", "funnel", "acquisition", "operational", "revenue"]}
    return Inputs(scored.items, {h.id: h for h in store.list(Hypothesis)}, store.list(Recommendation), counts,
                  scored.checks)


def payload(candidates: list[Opportunity], hypotheses: dict[str, Hypothesis]) -> dict[str, str]:
    items = []
    for o in candidates:
        h = hypotheses.get(o.hypothesis_id or "")
        items.append({"id": o.id, "title": o.title, "category": o.category,
                      "problem": h.observed_problem if h else o.description,
                      "intervention": h.intervention if h else o.description,
                      "linked_signals": o.signal_ids})
    return {"candidates": json.dumps(items, indent=2)}


@dataclass
class ProposalResult:
    proposal: Optional[OrchestratorProposal]
    ignored_ids: list[str]
    error: str = ""

    @property
    def tags(self) -> dict[str, ItemTags]:
        return {t.item_id: t for t in self.proposal.items} if self.proposal else {}


def propose(candidates: list[Opportunity], hypotheses: dict[str, Hypothesis], client: Any = None) -> ProposalResult:
    """LLM proposal, filtered to known candidates. Unknown ids are dropped and reported; clusters
    left with fewer than two known members are discarded."""
    try:
        raw = complete_json(PROMPT, OrchestratorProposal, payload(candidates, hypotheses), client=client)
    except LLMError as exc:
        return ProposalResult(None, [], str(exc))
    known = {o.id for o in candidates}
    ignored = sorted({i for c in raw.clusters for i in c.member_ids if i not in known}
                     | {t.item_id for t in raw.items if t.item_id not in known})
    clusters = []
    for c in raw.clusters:
        members = [i for i in dict.fromkeys(c.member_ids) if i in known]
        if len(members) >= 2:
            clusters.append(c.model_copy(update={"member_ids": members}))
    items = [t for t in raw.items if t.item_id in known]
    return ProposalResult(OrchestratorProposal(clusters=clusters, items=items), ignored)


def apply_tags(items: list[Opportunity], tags: dict[str, ItemTags]) -> list[Opportunity]:
    out = []
    for o in items:
        o = o.model_copy(deep=True)
        t = tags.get(o.id)
        if t:
            o.dependencies = list(dict.fromkeys(o.dependencies + list(t.dependencies)))
            o.expected_learning = o.expected_learning or t.expected_learning
            o.primary_metric = o.primary_metric or t.primary_metric
        out.append(o)
    return out


@dataclass
class Roadmap:
    items: list[Opportunity]
    balance: analysis.Balance
    hypotheses: dict[str, Hypothesis]
    recommendations: list[Recommendation]
    merges: list[tuple[list[str], Optional[str]]]
    now_capacity: Optional[int] = None
    checks: dict[str, Any] = field(default_factory=dict)

    @property
    def active(self) -> list[Opportunity]:
        return analysis.active(self.items)

    def lane(self, horizon: str) -> list[Opportunity]:
        return sorted([o for o in self.active if o.horizon == horizon], key=lambda o: o.horizon_order or 0)


def build(inputs: Inputs, proposal: ProposalResult, merges: list[tuple[list[str], Optional[str]]],
          now_capacity: Optional[int] = 3) -> Roadmap:
    items = apply_tags(inputs.candidates, proposal.tags)
    items = analysis.merge(items, merges)
    items = analysis.assign_defaults(items)
    return Roadmap(items, analysis.balance(items, now_capacity), inputs.hypotheses, inputs.recommendations,
                   merges, now_capacity, inputs.checks)


def rebalance(roadmap: Roadmap) -> Roadmap:
    """Recompute ordering and balance after user edits."""
    items = analysis.assign_defaults(roadmap.items)
    return Roadmap(items, analysis.balance(items, roadmap.now_capacity), roadmap.hypotheses,
                   roadmap.recommendations, roadmap.merges, roadmap.now_capacity, roadmap.checks)


def save(roadmap: Roadmap, store: Store) -> list[str]:
    """Persist roadmap fields on the Opportunity records (new ones belong to the scorer backlog)."""
    ids = []
    for o in roadmap.items:
        module = None if store.module_of(Opportunity, o.id) is not None else "experiment_opportunity_scorer"
        ids.append(store.save(o, module=module))
    return ids
