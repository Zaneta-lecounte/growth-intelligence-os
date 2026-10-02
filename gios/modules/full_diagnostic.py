"""Run Full Diagnostic: Signal → Diagnosis → Validation → Scoring → Orchestration in one pass on the
synthetic data, then check deterministically that the embedded stories surfaced and the red
herring did not."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from gios.core.schemas import Hypothesis, Opportunity, Recommendation, Signal
from gios.core.store import Store
from gios.modules import demo_flow

STEPS = [
    ("signals", "Signal layer", "Customer Signal Synthesizer, Behavioral Friction Analyzer, Leakage Auditor"),
    ("diagnostics", "Diagnosis", "Growth Intelligence Diagnostic for all channels and for paid search"),
    ("validations", "Validation", "Hypothesis Evidence Validator scores every hypothesis"),
    ("backlog", "Scoring", "Experiment Opportunity Scorer builds and scores the backlog"),
    ("roadmap", "Orchestration", "Growth Priority Orchestrator merges duplicates and schedules Now / Next / Later"),
]
RUNNERS = {
    "signals": demo_flow.seed_signals, "diagnostics": demo_flow.seed_diagnostics,
    "validations": demo_flow.seed_validations, "backlog": lambda store, client=None: demo_flow.seed_backlog(store),
    "roadmap": demo_flow.seed_roadmap,
}

Predicate = Callable[[dict], bool]
# Each embedded story and the evidence that identifies it (by signal attributes, not ids).
STORIES: dict[str, tuple[str, list[tuple[str, Predicate]]]] = {
    "pricing": ("Paid search → pricing page", [
        ("pricing-clarity verbatims", lambda a: a.get("theme") == "pricing_uncertainty"),
        ("pricing-page exits for paid search",
         lambda a: a.get("page") == "pricing" and a.get("channel") == "paid_search"),
        ("rising paid search CAC", lambda a: a.get("kind") == "rising_cac" and a.get("source") == "paid_search"),
    ]),
    "webinar": ("Webinar qualification mismatch", [
        ("webinar MQL→SQL leak",
         lambda a: a.get("kind") == "leak" and a.get("source") == "webinar" and a.get("stage") == "MQL→SQL"),
        ("webinar qualification mismatch flag",
         lambda a: a.get("kind") == "qualification_mismatch" and a.get("source") == "webinar"),
    ]),
    "mobile": ("Mobile demo form friction", [
        ("mobile demo-form friction", lambda a: a.get("page") == "demo" and a.get("device") == "mobile"),
        ("technical friction", lambda a: a.get("page") == "demo" and a.get("friction_type") == "technical"),
    ]),
}
RED_HERRING_THEME = "usability_polish"


@dataclass
class StoryCoverage:
    key: str
    name: str
    item: Optional[Opportunity]
    matched: list[str]
    required: list[str]

    @property
    def complete(self) -> bool:
        return self.item is not None and len(self.matched) == len(self.required)

    @property
    def prioritized(self) -> bool:
        return self.complete and self.item.horizon == "now"


@dataclass
class RedHerring:
    signal: Optional[Signal]
    on_roadmap: list[str]

    @property
    def frequency_rank(self) -> Optional[int]:
        return int(self.signal.attributes["frequency_rank"]) if self.signal else None

    @property
    def overall_rank(self) -> Optional[int]:
        return int(self.signal.attributes["overall_rank"]) if self.signal else None

    @property
    def deprioritized(self) -> bool:
        return not self.on_roadmap


@dataclass
class FullRun:
    counts: dict[str, int]
    stories: list[StoryCoverage]
    red_herring: RedHerring
    roadmap: Any = None
    errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(s.prioritized for s in self.stories) and self.red_herring.deprioritized


def reset_working_records(store: Store) -> None:
    """Clear the records the full run regenerates; experiments and learnings are kept."""
    for model in (Signal, Hypothesis, Opportunity, Recommendation):
        store.clear(model)


def current_roadmap(store: Store):
    from gios.modules.growth_priority_orchestrator import pipeline as gpo

    inputs = gpo.gather(store)
    return gpo.build(inputs, gpo.ProposalResult(None, []), merges=[])


def coverage(store: Store, roadmap) -> tuple[list[StoryCoverage], RedHerring]:
    signals = {s.id: s for s in store.list(Signal)}
    active = [o for o in roadmap.active if o.horizon]
    stories = []
    for key, (name, predicates) in STORIES.items():
        best, best_matched = None, []
        for o in sorted(active, key=lambda o: ("now", "next", "later").index(o.horizon)):
            attrs = [signals[i].attributes for i in o.signal_ids if i in signals]
            matched = [label for label, pred in predicates if any(pred(a) for a in attrs)]
            if len(matched) > len(best_matched):
                best, best_matched = o, matched
        stories.append(StoryCoverage(key, name, best, best_matched, [label for label, _ in predicates]))
    herring = next((s for s in signals.values() if s.attributes.get("theme") == RED_HERRING_THEME), None)
    on_roadmap = [o.title for o in active if herring and herring.id in o.signal_ids]
    return stories, RedHerring(herring, on_roadmap)


def run(store: Store, on_step: Optional[Callable[[int, str, str, int], None]] = None,
        client: Any = None) -> FullRun:
    reset_working_records(store)
    counts = {}
    for i, (key, label, _) in enumerate(STEPS):
        counts[key] = RUNNERS[key](store, client) if key != "backlog" else RUNNERS[key](store)
        if on_step:
            on_step(i, key, label, counts[key])
    roadmap = current_roadmap(store)
    stories, herring = coverage(store, roadmap)
    return FullRun(counts, stories, herring, roadmap)


def latest(store: Store) -> Optional[FullRun]:
    """Rebuild the result view from the store if a roadmap already exists."""
    if not any(o.horizon for o in store.list(Opportunity)):
        return None
    roadmap = current_roadmap(store)
    stories, herring = coverage(store, roadmap)
    counts = {"signals": len(store.list(Signal)), "diagnostics": len(store.list(Hypothesis)),
              "backlog": len(store.list(Opportunity)), "roadmap": len(roadmap.active)}
    return FullRun(counts, stories, herring, roadmap)
