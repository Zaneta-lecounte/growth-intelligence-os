"""Seed the GIOS store end to end with default settings, so any page can be explored
without first clicking through every upstream module. Each step is the real module code.

signal layer -> diagnostic (all channels, paid search) -> validator -> opportunity backlog
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


def diagnostic_inputs(scope: str):
    from gios.modules.growth_intelligence_diagnostic import BusinessSignal, Filters
    from gios.modules.growth_intelligence_diagnostic import analysis as gid

    channel, metric = DIAGNOSTIC_SCOPES[scope]
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


def seed_through(step: str, store: Optional[Store] = None, client: Any = None) -> dict[str, int]:
    """Run every step up to and including `step` (signals | diagnostics | validations | backlog)."""
    steps = ["signals", "diagnostics", "validations", "backlog"]
    store = store or Store()
    out: dict[str, int] = {}
    for name in steps[: steps.index(step) + 1]:
        if name == "signals":
            out[name] = seed_signals(store, client)
        elif name == "diagnostics":
            out[name] = seed_diagnostics(store, client)
        elif name == "validations":
            out[name] = seed_validations(store, client)
        else:
            out[name] = seed_backlog(store)
    return out
