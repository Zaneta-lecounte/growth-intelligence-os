"""Deterministic parts of the Hypothesis Evidence Validator: the six-part standard, the
evidence map, evidence-based score caps, total, band, recommendation, and weakest area."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from gios.core.schemas import HYPOTHESIS_PARTS, Hypothesis, Signal
from gios.modules.hypothesis_evidence_validator.models import DIMENSIONS, ValidatorProposal

MODULE = "hypothesis_evidence_validator"
SIGNAL_TYPES = ["customer", "behavioral", "funnel", "acquisition", "operational", "revenue"]
PART_LABELS = {
    "observed_problem": "Observed problem", "affected_audience": "Affected audience",
    "causal_explanation": "Proposed causal explanation", "intervention": "Proposed intervention",
    "expected_behavior_change": "Expected behavioral change", "expected_business_outcome": "Expected business outcome",
}
MIN_WORDS = 3
OUTCOME_TYPES = {"funnel", "revenue", "acquisition"}
MEASURABLE_TYPES = OUTCOME_TYPES | {"behavioral"}
RECOMMENDATION_LABELS = {"test": "Test", "research_first": "Research first",
                         "instrument_first": "Instrument first", "reject": "Reject"}
BAND_LABELS = {"do_not_test": "0–4: do not test yet", "research_or_instrument_first": "5–8: research / instrumentation first",
               "test_ready": "9–12: test-ready"}


# --- Six-part standard ---------------------------------------------------------------------


@dataclass
class PartCheck:
    part: str
    label: str
    text: str
    status: str  # ok | missing | vague

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def check_standard(h: Hypothesis) -> list[PartCheck]:
    checks = []
    for part in HYPOTHESIS_PARTS:
        text = getattr(h, part).strip()
        status = "missing" if not text else "vague" if len(text.split()) < MIN_WORDS else "ok"
        checks.append(PartCheck(part, PART_LABELS[part], text, status))
    return checks


# --- Evidence map -----------------------------------------------------------------------------


@dataclass
class EvidenceMap:
    supporting: dict[str, list[Signal]]
    contradicting: dict[str, list[Signal]]
    unknown_ids: list[str]

    @property
    def supporting_types(self) -> set[str]:
        return {t for t, items in self.supporting.items() if items}

    @property
    def n_supporting(self) -> int:
        return sum(len(v) for v in self.supporting.values())


def evidence_map(h: Hypothesis, signals: list[Signal]) -> EvidenceMap:
    """Group the hypothesis's linked Signal ids by signal type. Ids not in the store are unknown."""
    by_id = {s.id: s for s in signals}
    sup = {t: [] for t in SIGNAL_TYPES}
    con = {t: [] for t in SIGNAL_TYPES}
    unknown = []
    for ids, bucket in ((h.signal_ids, sup), (h.contradicting_signal_ids, con)):
        for i in dict.fromkeys(ids):
            if i in by_id:
                bucket[by_id[i].type].append(by_id[i])
            else:
                unknown.append(i)
    return EvidenceMap(sup, con, list(dict.fromkeys(unknown)))


# --- Caps -------------------------------------------------------------------------------------


def evidence_caps(em: EvidenceMap, standard: list[PartCheck]) -> dict[str, tuple[int, str]]:
    """Maximum defensible score per dimension given the linked evidence (rules, not judgement).
    An assumption-only hypothesis (no linked signals) can score at most 4."""
    types = em.supporting_types
    missing = {c.part for c in standard if not c.ok}
    caps: dict[str, tuple[int, str]] = {}
    caps["evidence_diversity"] = ((0, "no linked signals") if not types else
                                  (1, "signals of only one type") if len(types) == 1 else (2, ""))
    caps["behavioral_support"] = (2, "") if "behavioral" in types else (0, "no behavioral signal linked")
    caps["customer_support"] = (2, "") if "customer" in types else (0, "no customer signal linked")
    caps["business_relevance"] = ((2, "") if types & OUTCOME_TYPES else
                                  (1, "no funnel, acquisition or revenue signal ties it to an outcome"))
    vague_test = {"intervention", "expected_behavior_change"} & missing
    caps["testability"] = ((0, "intervention and expected behavior change are missing or vague")
                           if len(vague_test) == 2 else
                           (1, "intervention or expected behavior change is missing or vague") if vague_test else (2, ""))
    caps["measurement_readiness"] = ((2, "") if types & MEASURABLE_TYPES else
                                     (1, "no measured baseline in a linked funnel, behavioral, acquisition or "
                                         "revenue signal"))
    return caps


@dataclass
class DimensionRow:
    dimension: str
    proposed: Optional[int]
    cap: int
    cap_reason: str
    score: int
    justification: str
    overridden: bool = False

    @property
    def capped(self) -> bool:
        return self.proposed is not None and self.proposed > self.cap

    @property
    def above_cap(self) -> bool:
        return self.score > self.cap


def score_rows(proposal: Optional[ValidatorProposal], caps: dict[str, tuple[int, str]],
               overrides: Optional[dict[str, int]] = None) -> list[DimensionRow]:
    """Final score = user override if given, else min(LLM proposal, evidence cap), else 0."""
    overrides = overrides or {}
    proposed = proposal.by_dimension() if proposal else {}
    rows = []
    for dim in DIMENSIONS:
        cap, reason = caps[dim]
        p = proposed.get(dim)
        default = min(p.score, cap) if p else 0
        score = overrides.get(dim, default)
        if not 0 <= score <= 2:
            raise ValueError(f"{dim} score must be 0–2")
        rows.append(DimensionRow(dim, p.score if p else None, cap, reason, score,
                                 p.justification if p else "No proposal; scored manually.",
                                 overridden=dim in overrides and overrides[dim] != default))
    return rows


# --- Total, band, recommendation ----------------------------------------------------------------


def total(rows: list[DimensionRow]) -> int:
    return sum(r.score for r in rows)


def band(total_score: int) -> str:
    return "do_not_test" if total_score <= 4 else "research_or_instrument_first" if total_score <= 8 else "test_ready"


def recommendation(rows: list[DimensionRow]) -> str:
    """9–12 Test; 5–8 Instrument first when measurement readiness is the binding gap, else
    Research first; 0–4 Reject (not test-ready; rework with evidence)."""
    s = {r.dimension: r.score for r in rows}
    t = total(rows)
    if t >= 9:
        return "test"
    if t >= 5:
        evidence = min(s["evidence_diversity"], s["behavioral_support"], s["customer_support"])
        if s["measurement_readiness"] < 2 and s["measurement_readiness"] <= evidence:
            return "instrument_first"
        return "research_first"
    return "reject"


def weakest(rows: list[DimensionRow]) -> DimensionRow:
    return min(rows, key=lambda r: (r.score, DIMENSIONS.index(r.dimension)))


def deterministic_gaps(standard: list[PartCheck], em: EvidenceMap) -> list[str]:
    gaps = [f"{c.label} is {c.status}." for c in standard if not c.ok]
    if not em.n_supporting:
        gaps.append("No signals are linked: this hypothesis rests on assumption only.")
    for t, label in (("customer", "customer"), ("behavioral", "behavioral")):
        if t not in em.supporting_types:
            gaps.append(f"No {label} signal supports it.")
    if not em.supporting_types & OUTCOME_TYPES:
        gaps.append("No funnel, acquisition or revenue signal ties it to a business outcome.")
    if em.unknown_ids:
        gaps.append("Linked ids not found in the store: " + ", ".join(em.unknown_ids))
    return gaps
