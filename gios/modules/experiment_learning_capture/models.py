"""LLM output model for Experiment Learning Capture (narrative sections + tags)."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from gios.core.report import Narrative
from gios.modules.customer_signal_synthesizer.models import Theme


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NextHypothesis(_Strict):
    title: Narrative
    observed_problem: Narrative
    affected_audience: Narrative
    causal_explanation: Narrative
    intervention: Narrative
    expected_behavior_change: Narrative
    expected_business_outcome: Narrative


class LearningDraft(_Strict):
    theme: Theme = Field(description="Customer-problem theme this experiment addressed.")
    original_problem: Narrative
    what_happened: Narrative
    learned_about_customer: Narrative
    learned_about_journey: Narrative
    learned_about_business: Narrative
    should_not_conclude: Narrative = Field(min_length=10)
    reusable_principle: Narrative
    next_hypothesis: NextHypothesis
