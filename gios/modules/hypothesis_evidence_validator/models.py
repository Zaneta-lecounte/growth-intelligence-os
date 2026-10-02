"""LLM output model for the Hypothesis Evidence Validator."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gios.core.report import Narrative
from gios.core.schemas import Hypothesis

Dimension = Literal["evidence_diversity", "behavioral_support", "customer_support",
                    "business_relevance", "testability", "measurement_readiness"]
DIMENSIONS: tuple[str, ...] = Hypothesis.SCORE_FIELDS
DIMENSION_LABELS = {
    "evidence_diversity": "Evidence diversity", "behavioral_support": "Behavioral support",
    "customer_support": "Customer support", "business_relevance": "Business relevance",
    "testability": "Testability", "measurement_readiness": "Measurement readiness",
}
# hypothesis-evidence-validator.md, Validation dimensions
RUBRIC = {
    "evidence_diversity": ["one unsupported assumption", "one credible signal", "multiple independent signal types"],
    "behavioral_support": ["none", "indirect", "direct"],
    "customer_support": ["none", "anecdotal", "recurring / well-supported"],
    "business_relevance": ["weak", "moderate", "directly tied to target outcome"],
    "testability": ["vague", "partially measurable", "clear causal test"],
    "measurement_readiness": ["cannot measure", "partial", "reliable"],
}


class DimensionScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension: Dimension
    score: int = Field(ge=0, le=2)
    justification: Narrative = Field(max_length=240, description="One line.")


class ValidatorProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scores: list[DimensionScore]
    missing_evidence: list[Narrative] = Field(default_factory=list)

    @model_validator(mode="after")
    def _one_score_per_dimension(self):
        dims = [s.dimension for s in self.scores]
        if sorted(dims) != sorted(DIMENSIONS):
            raise ValueError(f"provide exactly one score for each of: {', '.join(DIMENSIONS)}")
        return self

    def by_dimension(self) -> dict[str, DimensionScore]:
        return {s.dimension: s for s in self.scores}
