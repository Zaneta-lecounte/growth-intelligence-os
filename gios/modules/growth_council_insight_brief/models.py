"""LLM output model for the Growth Council Insight Brief."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gios.core.report import Narrative
from gios.modules.qualified_demand_leakage_auditor.models import OwnerCategory


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SignalLine(_Strict):
    text: Narrative
    ids: list[str] = Field(default_factory=list, description="Ids of the stored records this line rests on.")


class BriefSignals(_Strict):
    customer: SignalLine
    behavioral: SignalLine
    funnel: SignalLine
    revenue: SignalLine
    sales_operational: SignalLine


class BriefDraft(_Strict):
    what_changed: Narrative
    why_it_matters: Narrative
    signals: BriefSignals
    working_hypothesis: Narrative
    hypothesis_id: Optional[str] = None
    recommended_action: Narrative
    opportunity_id: Optional[str] = None
    cross_functional_dependency: Narrative
    owner: OwnerCategory
    owner_rationale: Narrative
    measurement: Narrative
    organization_learning: Narrative
    learning_ids: list[str] = Field(default_factory=list)
    decision_needed: Narrative

    def cited_ids(self) -> set[str]:
        ids = {i for line in self.signals.model_dump().values() for i in line["ids"]}
        ids |= set(self.learning_ids)
        ids |= {i for i in (self.hypothesis_id, self.opportunity_id) if i}
        return ids


def checked_brief_model(allowed_ids: frozenset[str]) -> type[BriefDraft]:
    """BriefDraft that rejects citations of records not in the selected period's inputs."""

    class CheckedBriefDraft(BriefDraft):
        @model_validator(mode="after")
        def _known_ids(self):
            unknown = sorted(self.cited_ids() - allowed_ids)
            if unknown:
                raise ValueError(f"brief cites records that are not in the provided inputs: {unknown}")
            return self

    return CheckedBriefDraft
