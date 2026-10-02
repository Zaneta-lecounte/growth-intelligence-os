"""LLM output models for the Behavioral Friction Analyzer."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from gios.core.report import Narrative

# behavioral-friction-analyzer.md, Friction taxonomy
FrictionType = Literal[
    "comprehension", "motivation", "trust", "effort", "navigation", "technical",
    "expectation_mismatch", "decision_overload", "handoff",
]
FRICTION_LABELS = {
    "comprehension": "Comprehension friction", "motivation": "Motivation friction",
    "trust": "Trust friction", "effort": "Effort friction", "navigation": "Navigation friction",
    "technical": "Technical friction", "expectation_mismatch": "Expectation mismatch",
    "decision_overload": "Decision overload", "handoff": "Handoff friction",
}
# behavioral-friction-analyzer.md, Step 4
Action = Literal["qualitative_research", "ux_experiment", "messaging_experiment", "technical_fix",
                 "instrumentation_improvement"]
ACTION_LABELS = {
    "qualitative_research": "Qualitative research", "ux_experiment": "UX experiment",
    "messaging_experiment": "Messaging experiment", "technical_fix": "Technical fix",
    "instrumentation_improvement": "Instrumentation improvement",
}


class FrictionClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    finding_id: str = Field(description="The finding id, copied exactly.")
    friction_type: FrictionType
    what_happened: Narrative = Field(description="Plain description of the behavior, no causal claim.")
    competing_explanations: list[Narrative] = Field(min_length=2)
    validation_needed: list[Narrative] = Field(min_length=1, description="Customer evidence or data that would strengthen or refute the diagnosis.")
    supporting_customer_signal_ids: list[str] = Field(
        default_factory=list, description="Ids of provided customer signals that support a cause. Empty if none.")
    recommended_action: Action
    recommended_detail: Narrative


class FrictionBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classifications: list[FrictionClassification]
