"""Input and LLM output models for the Growth Intelligence Diagnostic."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gios.core.report import Narrative
from gios.core.schemas import ActionType

ACTION_LABELS = {
    "experiment": "Experiment", "research": "Additional research",
    "instrumentation_fix": "Analytics / instrumentation fix", "messaging_change": "Messaging change",
    "journey_redesign": "Journey redesign", "targeting_change": "Targeting change",
    "process_change": "Process / handoff change",
}


class BusinessSignal(BaseModel):
    """Step 1: the business signal must be defined before any diagnosis."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    goal: str = Field(min_length=3, description="Business goal")
    target_metric: str = Field(min_length=2)
    what: str = Field(min_length=3, description="What is underperforming")
    baseline: str = Field(min_length=3, description="Compared with what baseline")
    segment: str = Field(min_length=2, description="For which audience / segment")
    period: str = Field(min_length=3, description="Over what period")


STEP1_FIELDS = {"goal": "Business goal", "target_metric": "Target metric", "what": "What is underperforming",
                "baseline": "Baseline", "segment": "Audience / segment", "period": "Period"}


def missing_business_fields(values: dict) -> list[str]:
    """Labels of Step 1 fields that are not yet filled in (min lengths from BusinessSignal)."""
    missing = []
    for name, label in STEP1_FIELDS.items():
        min_len = BusinessSignal.model_fields[name].metadata[0].min_length
        if len(str(values.get(name) or "").strip()) < min_len:
            missing.append(label)
    return missing


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceClaim(_Strict):
    signal_id: str = Field(description="Id of a provided signal, copied exactly.")
    claim: Narrative
    status: Literal["observed", "inferred"] = Field(
        description="observed = the signal directly shows this; inferred = your interpretation.")


class RootCause(_Strict):
    title: Narrative
    observed_problem: Narrative
    affected_audience: Narrative
    causal_explanation: Narrative = Field(description="Suspected cause")
    intervention: Narrative
    expected_behavior_change: Narrative
    expected_business_outcome: Narrative
    supporting: list[EvidenceClaim] = Field(min_length=1)
    contradicting: list[EvidenceClaim] = Field(default_factory=list)
    missing: list[Narrative] = Field(default_factory=list, description="Evidence we do not have.")


class DiagnosisOutput(_Strict):
    growth_problem: Narrative
    hypotheses: list[RootCause] = Field(min_length=2, max_length=5)
    next_best_action: ActionType
    action_detail: Narrative
    primary_metric: Narrative
    guardrail_metric: Narrative
    downstream_metric: Narrative
    learning_objective: Narrative
    missing_evidence: list[Narrative] = Field(default_factory=list)

    def cited_ids(self) -> set[str]:
        return {c.signal_id for h in self.hypotheses for c in h.supporting + h.contradicting}


def checked_output_model(allowed_ids: frozenset[str]) -> type[DiagnosisOutput]:
    """DiagnosisOutput that rejects any evidence claim citing a signal id not in `allowed_ids`.
    A rejection triggers llm.py's single retry with the error, then fails."""

    class CheckedDiagnosisOutput(DiagnosisOutput):
        @model_validator(mode="after")
        def _known_signal_ids(self):
            unknown = sorted(self.cited_ids() - allowed_ids)
            if unknown:
                raise ValueError(f"evidence cites signal ids that do not exist in the provided signals: {unknown}")
            return self

    return CheckedDiagnosisOutput
