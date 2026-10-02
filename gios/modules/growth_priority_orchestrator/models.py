"""LLM output model for the Growth Priority Orchestrator (duplicate clusters + qualitative tags)."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from gios.core.report import Narrative
from gios.core.schemas import Dependency


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Cluster(_Strict):
    member_ids: list[str] = Field(min_length=2, description="Ids of candidates that describe the same opportunity.")
    canonical_title: Narrative
    rationale: Narrative


class ItemTags(_Strict):
    item_id: str
    dependencies: list[Dependency] = Field(default_factory=list)
    expected_learning: Narrative
    primary_metric: Narrative


class OrchestratorProposal(_Strict):
    clusters: list[Cluster] = Field(default_factory=list)
    items: list[ItemTags] = Field(default_factory=list)
