"""Shared Pydantic models passed between GIOS modules.

Vocabularies (Literal types) mirror the specs in specs/. Derived numbers (GROWTH priority
score, hypothesis validation total/verdict) are computed here deterministically as plain
properties, never supplied by the LLM. Properties are not serialized, so models round-trip
cleanly through the store with extra="forbid".
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

# --- Shared vocabularies -------------------------------------------------------------------

# growth-intelligence-diagnostic.md, Step 2
SignalType = Literal["customer", "behavioral", "funnel", "acquisition", "operational", "revenue"]
# growth-intelligence-diagnostic.md, Step 3
EvidenceStatus = Literal["observed", "inferred", "unknown"]
Segment = Literal["smb", "mid_market", "all"]
JourneyStage = Literal[
    "awareness", "consideration", "evaluation", "purchase", "onboarding", "retention", "unknown"
]
Confidence = Literal["low", "medium", "high"]
# growth-priority-orchestrator.md, Step 2
OpportunityCategory = Literal[
    "customer_problem", "acquisition", "conversion", "activation",
    "qualification", "retention", "revenue", "operations_measurement",
]
# experiment-opportunity-scorer.md, Recommendation
OpportunityRecommendation = Literal["run_now", "research_first", "instrument_first", "defer", "reject"]
# experiment-opportunity-scorer.md, Guardrails
RiskFlag = Literal[
    "strategic_dependency", "measurement_readiness", "legal_privacy",
    "brand", "sample_size", "implementation_dependency",
]
# hypothesis-evidence-validator.md, Total score
HypothesisVerdict = Literal["do_not_test", "research_or_instrument_first", "test_ready"]
ExperimentStatus = Literal["planned", "running", "completed", "stopped"]
# downstream-impact-analyzer.md / experiment-learning-capture.md
Decision = Literal["scale", "iterate", "retest", "stop", "observe_longer", "research"]
# growth-intelligence-diagnostic.md, Step 6
ActionType = Literal[
    "experiment", "research", "instrumentation_fix", "messaging_change",
    "journey_redesign", "targeting_change", "process_change",
]
Horizon = Literal["now", "next", "later"]



def _score(lo: int, hi: int, optional: bool = True):
    return Field(default=None, ge=lo, le=hi) if optional else Field(ge=lo, le=hi)


def _new_id() -> str:
    return uuid4().hex[:12]


def _now() -> datetime:
    return datetime.now(timezone.utc)


class GIOSModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str = Field(default_factory=_new_id)
    created_at: datetime = Field(default_factory=_now)


# --- Signal layer --------------------------------------------------------------------------


class Signal(GIOSModel):
    type: SignalType
    evidence_status: EvidenceStatus
    source: str = Field(min_length=1, description="Data source, e.g. 'web_behavior.csv' or 'interview'.")
    segment: Segment
    journey_stage: JourneyStage
    summary: str = Field(min_length=1)
    strength: int = _score(1, 5, optional=False)


# --- Diagnosis / prioritization layer ------------------------------------------------------


class Opportunity(GIOSModel):
    title: str = Field(min_length=1)
    description: str = ""
    category: OpportunityCategory
    segment: Segment = "all"
    signal_ids: list[str] = Field(default_factory=list)

    # GROWTH dimensions (experiment-opportunity-scorer.md); T: 1 = very low effort.
    growth_impact: Optional[int] = _score(1, 5)
    research_evidence: Optional[int] = _score(1, 5)
    opportunity_size: Optional[int] = _score(1, 5)
    web_evidence: Optional[int] = _score(1, 5)
    technical_effort: Optional[int] = _score(1, 5)
    hypothesis_confidence: Optional[int] = _score(1, 5)

    risk_flags: list[RiskFlag] = Field(default_factory=list)
    key_risk: str = ""
    recommendation: Optional[OpportunityRecommendation] = None

    @property
    def is_scored(self) -> bool:
        return None not in (
            self.growth_impact, self.research_evidence, self.opportunity_size,
            self.web_evidence, self.technical_effort, self.hypothesis_confidence,
        )

    @property
    def priority_score(self) -> Optional[float]:
        """Priority Score = (G × R × O × W × H) / T, or None until all six are scored."""
        if not self.is_scored:
            return None
        numerator = (
            self.growth_impact * self.research_evidence * self.opportunity_size
            * self.web_evidence * self.hypothesis_confidence
        )
        return round(numerator / self.technical_effort, 2)


class Hypothesis(GIOSModel):
    opportunity_id: Optional[str] = None
    # Hypothesis standard (hypothesis-evidence-validator.md)
    observed_problem: str = Field(min_length=1)
    affected_audience: str = Field(min_length=1)
    causal_explanation: str = Field(min_length=1)
    intervention: str = Field(min_length=1)
    expected_behavior_change: str = Field(min_length=1)
    expected_business_outcome: str = Field(min_length=1)
    signal_ids: list[str] = Field(default_factory=list)

    # Validation dimensions, 0–2 each
    evidence_diversity: Optional[int] = _score(0, 2)
    behavioral_support: Optional[int] = _score(0, 2)
    customer_support: Optional[int] = _score(0, 2)
    business_relevance: Optional[int] = _score(0, 2)
    testability: Optional[int] = _score(0, 2)
    measurement_readiness: Optional[int] = _score(0, 2)

    def _dimension_scores(self) -> list[Optional[int]]:
        return [
            self.evidence_diversity, self.behavioral_support, self.customer_support,
            self.business_relevance, self.testability, self.measurement_readiness,
        ]

    @property
    def validation_total(self) -> Optional[int]:
        scores = self._dimension_scores()
        return None if None in scores else sum(scores)

    @property
    def verdict(self) -> Optional[HypothesisVerdict]:
        """0–4 do not test yet; 5–8 research / instrumentation first; 9–12 test-ready."""
        total = self.validation_total
        if total is None:
            return None
        if total <= 4:
            return "do_not_test"
        if total <= 8:
            return "research_or_instrument_first"
        return "test_ready"


# --- Learning layer ------------------------------------------------------------------------


class Variant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    visitors: int = Field(ge=0)
    conversions: int = Field(ge=0)
    mqls: int = Field(default=0, ge=0)
    sqls: int = Field(default=0, ge=0)
    opps: int = Field(default=0, ge=0)
    wins: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _funnel_is_monotonic(self) -> "Variant":
        chain = [self.visitors, self.conversions, self.mqls, self.sqls, self.opps, self.wins]
        if any(later > earlier for earlier, later in zip(chain, chain[1:])):
            raise ValueError("variant funnel counts must not increase down the funnel")
        return self

    @property
    def conversion_rate(self) -> float:
        return self.conversions / self.visitors if self.visitors else 0.0


class Experiment(GIOSModel):
    name: str = Field(min_length=1)
    hypothesis_id: Optional[str] = None
    hypothesis_text: str = ""
    primary_metric: str = "conversion_rate"
    guardrail_metrics: list[str] = Field(default_factory=list)
    variants: list[Variant] = Field(min_length=2)
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    status: ExperimentStatus = "planned"

    @model_validator(mode="after")
    def _dates_ordered(self) -> "Experiment":
        if self.start_date and self.end_date and self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")
        return self


class Learning(GIOSModel):
    experiment_id: Optional[str] = None
    original_problem: str = ""
    what_happened: str = Field(min_length=1)
    learned_about_customer: str = ""
    learned_about_journey: str = ""
    learned_about_business: str = ""
    should_not_conclude: str = ""
    decision: Decision
    reusable_principle: str = ""
    next_hypothesis: str = ""


class Recommendation(GIOSModel):
    title: str = Field(min_length=1)
    action_type: ActionType
    rationale: str = Field(min_length=1)
    owner: str = ""
    horizon: Horizon = "now"
    primary_metric: str = ""
    guardrail_metric: str = ""
    downstream_metric: str = ""
    confidence: Confidence = "medium"
    opportunity_id: Optional[str] = None
