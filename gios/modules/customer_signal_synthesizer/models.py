"""LLM input/output models for the Customer Signal Synthesizer."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from gios.core.report import Narrative
from gios.core.schemas import JourneyStage

# Spec taxonomy (customer-signal-synthesizer.md, step 2) plus three GIOS extensions so
# product defects, cosmetic requests, and off-topic evidence don't pollute spec themes.
Theme = Literal[
    "unclear_value", "trust_proof", "pricing_uncertainty", "implementation_risk",
    "feature_fit", "switching_cost", "complexity", "urgency", "competitive_comparison",
    "technical_issue", "usability_polish", "other",
]
THEME_LABELS: dict[str, str] = {
    "unclear_value": "Unclear value",
    "trust_proof": "Trust / proof",
    "pricing_uncertainty": "Pricing uncertainty",
    "implementation_risk": "Implementation risk",
    "feature_fit": "Feature fit",
    "switching_cost": "Switching cost",
    "complexity": "Complexity",
    "urgency": "Urgency",
    "competitive_comparison": "Competitive comparison",
    "technical_issue": "Technical issue",
    "usability_polish": "Usability polish",
    "other": "Other",
}
Sentiment = Literal["negative", "neutral", "positive"]
Relevance = Literal["low", "medium", "high"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceTag(_Strict):
    key: str = Field(description="The key of the verbatim being tagged, copied exactly.")
    theme: Theme
    topic: Narrative = Field(max_length=80, description="Short topic label, a few words.")
    sentiment: Sentiment
    segment: Literal["smb", "mid_market", "unknown"] = Field(
        description="Segment implied by the text itself; 'unknown' unless the text says so.")
    journey_stage: JourneyStage = Field(description="Journey stage implied by the text itself.")
    business_relevance: Relevance


class TagBatch(_Strict):
    tags: list[EvidenceTag]


class ThemeAssessment(_Strict):
    theme: Theme
    severity: int = Field(ge=1, le=5, description="How badly this blocks or hurts the customer.")
    commercial_relevance: int = Field(ge=1, le=5, description="How directly it affects revenue.")
    journey_relevance: int = Field(ge=1, le=5, description="How central the affected journey stage is to acquisition.")
    summary: Narrative
    expected_behavior: Narrative = Field(description="Expected behavioral manifestation in analytics.")
    funnel_stage: Narrative = Field(description="Funnel stage likely affected.")
    analytics_signal: Narrative = Field(description="Analytics signal that could validate the theme.")
    testable_question: Narrative


class ThemeSynthesis(_Strict):
    assessments: list[ThemeAssessment]
    tensions: list[Narrative] = Field(description="Customer tensions: competing needs or contradictions.")
    research_gaps: list[Narrative]
