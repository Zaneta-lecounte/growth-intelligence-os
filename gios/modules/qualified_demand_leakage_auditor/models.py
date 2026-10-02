"""LLM output model for the Qualified Demand Leakage Auditor (narrative only)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from gios.core.report import Narrative

OwnerCategory = Literal["acquisition", "web_conversion", "qualification", "operations_routing",
                        "sales_handoff", "product_offering", "measurement"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CauseNarrative(_Strict):
    owner: OwnerCategory
    explanation: Narrative


class LeakageNarrative(_Strict):
    about_leak: str = Field(description="The highest-value leak id you were given, copied exactly.")
    leakage_summary: Narrative
    probable_causes: list[CauseNarrative]
    recommended_intervention: Narrative
    primary_metric: Narrative
    downstream_metrics: list[Narrative]
