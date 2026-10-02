"""Hypothesis Evidence Validator pipeline: standard + evidence map + caps (code) ->
score proposal (LLM) -> overrides (user) -> total, band, recommendation (code)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from gios.core.llm import LLMError, complete_json
from gios.core.schemas import Hypothesis, Signal
from gios.modules.hypothesis_evidence_validator import analysis
from gios.modules.hypothesis_evidence_validator.analysis import DimensionRow, EvidenceMap, PartCheck
from gios.modules.hypothesis_evidence_validator.models import RUBRIC, ValidatorProposal

PROMPT = "validate_hypothesis"


@dataclass
class ValidationResult:
    hypothesis: Hypothesis
    standard: list[PartCheck]
    evidence: EvidenceMap
    proposal: Optional[ValidatorProposal]
    proposal_error: str
    rows: list[DimensionRow]

    @property
    def total(self) -> int:
        return analysis.total(self.rows)

    @property
    def band(self) -> str:
        return analysis.band(self.total)

    @property
    def recommendation(self) -> str:
        return analysis.recommendation(self.rows)

    @property
    def weakest(self) -> DimensionRow:
        return analysis.weakest(self.rows)

    @property
    def missing_evidence(self) -> list[str]:
        llm = list(self.proposal.missing_evidence) if self.proposal else []
        return list(dict.fromkeys(analysis.deterministic_gaps(self.standard, self.evidence) + llm))


def payload(h: Hypothesis, em: EvidenceMap, standard: list[PartCheck]) -> dict[str, str]:
    linked = [{"id": s.id, "relation": rel, "type": s.type, "evidence_status": s.evidence_status, "summary": s.summary}
              for rel, groups in (("supports", em.supporting), ("contradicts", em.contradicting))
              for items in groups.values() for s in items]
    return {
        "hypothesis": json.dumps({c.part: c.text or "(missing)" for c in standard}, indent=2),
        "rubric": json.dumps(RUBRIC, indent=2),
        "linked_signals": json.dumps(linked, indent=2) if linked else "None. No signals are linked to this hypothesis.",
    }


def propose(h: Hypothesis, em: EvidenceMap, standard: list[PartCheck],
            client: Any = None) -> tuple[Optional[ValidatorProposal], str]:
    try:
        return complete_json(PROMPT, ValidatorProposal, payload(h, em, standard), demo_key=h.fingerprint(),
                             client=client), ""
    except LLMError as exc:
        return None, str(exc)


def validate(h: Hypothesis, signals: list[Signal], overrides: Optional[dict[str, int]] = None,
             proposal: Optional[ValidatorProposal] = None, client: Any = None,
             auto_propose: bool = True, proposal_error: str = "") -> ValidationResult:
    standard = analysis.check_standard(h)
    em = analysis.evidence_map(h, signals)
    error = proposal_error
    if proposal is None and auto_propose:
        proposal, error = propose(h, em, standard, client)
    rows = analysis.score_rows(proposal, analysis.evidence_caps(em, standard), overrides)
    return ValidationResult(h, standard, em, proposal, error, rows)


def scored_hypothesis(result: ValidationResult) -> Hypothesis:
    """The hypothesis with the validator's final scores, justifications and recommendation."""
    h = result.hypothesis.model_copy(deep=True)
    for r in result.rows:
        setattr(h, r.dimension, r.score)
    h.score_justifications = {r.dimension: r.justification + (" (user override)" if r.overridden else "")
                              for r in result.rows}
    h.validator_recommendation = result.recommendation
    return h


def save(result: ValidationResult, store) -> str:
    """Write the score to the Hypothesis record (keeping its module; new ones belong to this module)."""
    h = scored_hypothesis(result)
    module = None if store.module_of(Hypothesis, h.id) is not None else analysis.MODULE
    return store.save(h, module=module)
