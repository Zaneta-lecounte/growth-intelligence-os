"""Seed the GIOS store end to end with default settings, so any page can be explored
without first clicking through every upstream module. Each step is the real module code.

signal layer -> diagnostic (all channels, paid search) -> validator -> opportunity backlog ->
roadmap (proposed duplicate merges confirmed) -> downstream impact of past experiments -> learnings
"""
from __future__ import annotations

from typing import Any, Optional

from gios.core import data
from gios.core.schemas import Hypothesis, Signal
from gios.core.store import Store

# Scopes with cached diagnoses in demo mode: scope -> (channel, metric for the Step 1 draft)
DIAGNOSTIC_SCOPES = {
    "all": (None, "MQL→SQL"),
    "paid_search": ("paid_search", "CAC"),
}
# Scopes with a cached diagnosis in demo/ (a superset of the seeded scopes).
CACHED_DIAGNOSTIC_SCOPES = {**DIAGNOSTIC_SCOPES, "webinar": ("webinar", "MQL→SQL")}


def diagnostic_inputs(scope: str):
    from gios.modules.growth_intelligence_diagnostic import BusinessSignal, Filters
    from gios.modules.growth_intelligence_diagnostic import analysis as gid

    channel, metric = CACHED_DIAGNOSTIC_SCOPES[scope]
    filters = Filters(channel=channel)
    draft = gid.suggest_business_signal(data.load("funnel_by_source"), metric, channel, None,
                                        filters.start, filters.end)
    return BusinessSignal(goal="Grow qualified pipeline efficiently", **draft), filters


def seed_signals(store: Store, client: Any = None) -> int:
    from gios.modules.signal_layer import run_signal_layer

    return sum(run_signal_layer(store, client).values())


def seed_diagnostics(store: Store, client: Any = None) -> int:
    from gios.modules.growth_intelligence_diagnostic import pipeline as gid

    signals = store.list(Signal)
    n = 0
    for scope in DIAGNOSTIC_SCOPES:
        business, filters = diagnostic_inputs(scope)
        result = gid.run(business, filters, signals, client=client)
        gid.save(result, store)
        n += len(result.ranked)
    return n


def seed_validations(store: Store, client: Any = None, only_unscored: bool = True) -> int:
    from gios.modules.hypothesis_evidence_validator import pipeline as hev

    signals = store.list(Signal)
    n = 0
    for h in store.list(Hypothesis):
        if only_unscored and h.validation_total is not None:
            continue
        hev.save(hev.validate(h, signals, client=client), store)
        n += 1
    return n


def seed_backlog(store: Store) -> int:
    from gios.modules.experiment_opportunity_scorer import pipeline as eos

    backlog = eos.build_backlog(store)
    eos.save(backlog, store)
    return len(backlog)


def seed_roadmap(store: Store, client: Any = None) -> int:
    from gios.modules.growth_priority_orchestrator import pipeline as gpo

    inputs = gpo.gather(store)
    proposal = gpo.propose(inputs.candidates, inputs.hypotheses, client)
    merges = [(c.member_ids, c.canonical_title) for c in proposal.proposal.clusters] if proposal.proposal else []
    roadmap = gpo.build(inputs, proposal, merges)
    gpo.save(roadmap, store)
    return len(roadmap.active)


def seed_experiments(store: Store) -> int:
    from gios.modules.downstream_impact_analyzer import analyze_all

    return len(analyze_all(store))


def seed_learnings(store: Store, client: Any = None) -> int:
    from gios.modules.experiment_learning_capture import draft_all

    return len(draft_all(store, client))


STEPS = ["signals", "diagnostics", "validations", "backlog", "roadmap", "experiments", "learnings"]


def seed_through(step: str, store: Optional[Store] = None, client: Any = None) -> dict[str, int]:
    """Run every step up to and including `step` (see STEPS)."""
    store = store or Store()
    runners = {
        "signals": lambda: seed_signals(store, client),
        "diagnostics": lambda: seed_diagnostics(store, client),
        "validations": lambda: seed_validations(store, client),
        "backlog": lambda: seed_backlog(store),
        "roadmap": lambda: seed_roadmap(store, client),
        "experiments": lambda: seed_experiments(store),
        "learnings": lambda: seed_learnings(store, client),
    }
    return {name: runners[name]() for name in STEPS[: STEPS.index(step) + 1]}
