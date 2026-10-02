"""Experiment Opportunity Scorer pipeline: backlog from stored hypotheses (+ manual rows) ->
prefills -> user edits -> score, sample-size flag, recommendation (all in code)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from gios.core import data
from gios.core.schemas import Hypothesis, Opportunity, Signal
from gios.core.store import Store
from gios.modules.experiment_opportunity_scorer import analysis
from gios.modules.experiment_opportunity_scorer.analysis import SampleCheck, Settings


def opportunity_id(hypothesis_id: str) -> str:
    return f"eos-{hypothesis_id}"


def from_hypothesis(h: Hypothesis, signals_by_id: dict[str, Signal]) -> Opportunity:
    linked = [signals_by_id[i] for i in h.signal_ids if i in signals_by_id]
    scores, _ = analysis.prefill(h, linked)
    return Opportunity(
        id=opportunity_id(h.id), title=h.label, description=h.intervention, hypothesis_id=h.id,
        category=analysis.category_for(linked), signal_ids=list(h.signal_ids),
        validator_total=h.validation_total, is_fix=analysis.is_fix(linked),
        **scores, **analysis.infer_population(h.signal_ids),
    )


def build_backlog(store: Store) -> list[Opportunity]:
    """Stored opportunities keep their edits; hypotheses without one get a prefilled row.
    Validator totals are refreshed from the hypothesis so H can follow later validation."""
    signals = {s.id: s for s in store.list(Signal)}
    hyps = {h.id: h for h in store.list(Hypothesis)}
    existing = {o.id: o for o in store.list(Opportunity, module=analysis.MODULE)}
    backlog = []
    for h in hyps.values():
        oid = opportunity_id(h.id)
        if oid in existing:
            o = existing.pop(oid)
            if o.validator_total != h.validation_total:
                o.validator_total = h.validation_total
                if h.validation_total is not None:
                    o.hypothesis_confidence = analysis.h_from_validator(h.validation_total)
            backlog.append(o)
        else:
            backlog.append(from_hypothesis(h, signals))
    backlog += list(existing.values())  # manual rows and rows whose hypothesis was removed
    return backlog


@dataclass
class ScoredBacklog:
    items: list[Opportunity]
    checks: dict[str, SampleCheck]
    rules: dict[str, str]
    settings: Settings
    notes: dict[str, dict[str, str]] = field(default_factory=dict)

    def ranked(self) -> list[Opportunity]:
        return sorted(self.items, key=lambda o: (-(o.priority_score or 0), o.title))


def score(backlog: list[Opportunity], settings: Settings = Settings(), store: Optional[Store] = None,
          web: Optional[pd.DataFrame] = None, funnel: Optional[pd.DataFrame] = None) -> ScoredBacklog:
    """Apply the sample-size auto-flag, recommendation rules and key risk to every row."""
    web = data.load("web_behavior") if web is None else web
    funnel = data.load("funnel_by_source") if funnel is None else funnel
    hyps = {h.id: h for h in store.list(Hypothesis)} if store else {}
    signals = {s.id: s for s in store.list(Signal)} if store else {}
    items, checks, rules, notes = [], {}, {}, {}
    for o in backlog:
        o = o.model_copy(deep=True)
        check = analysis.sample_check(o, web, funnel, settings)
        flags = [f for f in o.risk_flags if f != "sample_size"]
        if check.flagged:
            flags.append("sample_size")
        o.risk_flags = flags
        h = hyps.get(o.hypothesis_id or "")
        rec, rule = analysis.recommend(o, settings, h.validator_recommendation if h else None)
        o.recommendation = rec
        o.key_risk = analysis.key_risk(o, check, settings)
        items.append(o)
        checks[o.id], rules[o.id] = check, rule
        if h is not None:
            notes[o.id] = analysis.prefill(h, [signals[i] for i in h.signal_ids if i in signals])[1]
    return ScoredBacklog(items, checks, rules, settings, notes)


def save(backlog: list[Opportunity] | ScoredBacklog, store: Store) -> list[str]:
    items = backlog.items if isinstance(backlog, ScoredBacklog) else backlog
    keep = {o.id for o in items}
    for old in store.list(Opportunity, module=analysis.MODULE):
        if old.id not in keep:
            store.delete(Opportunity, old.id)
    return store.save_many(items, module=analysis.MODULE)
