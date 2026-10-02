"""Registry of GIOS modules. Each module's spec in specs/ is its source of truth."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from gios import config

Layer = Literal["signal", "diagnosis", "prioritization", "learning"]
Status = Literal["planned", "in_progress", "ready"]

LAYERS: dict[Layer, str] = {
    "signal": "Signal",
    "diagnosis": "Diagnosis",
    "prioritization": "Prioritization",
    "learning": "Learning",
}


@dataclass(frozen=True)
class ModuleInfo:
    slug: str
    name: str
    layer: Layer
    spec: str
    purpose: str
    status: Status = "planned"

    @property
    def spec_path(self):
        return config.SPECS_DIR / self.spec


MODULES: list[ModuleInfo] = [
    ModuleInfo("customer_signal_synthesizer", "Customer Signal Synthesizer", "signal",
               "customer-signal-synthesizer.md",
               "Turn qualitative customer evidence into structured themes without overstating frequency.",
               status="ready"),
    ModuleInfo("behavioral_friction_analyzer", "Behavioral Friction Analyzer", "signal",
               "behavioral-friction-analyzer.md",
               "Find where digital behavior signals confusion, hesitation, or abandonment.",
               status="ready"),
    ModuleInfo("growth_intelligence_diagnostic", "Growth Intelligence Diagnostic", "diagnosis",
               "growth-intelligence-diagnostic.md",
               "Identify the most important growth problem and the next best action."),
    ModuleInfo("qualified_demand_leakage_auditor", "Qualified Demand Leakage Auditor", "diagnosis",
               "qualified-demand-leakage-auditor.md",
               "Find where qualified demand is lost between acquisition and revenue."),
    ModuleInfo("hypothesis_evidence_validator", "Hypothesis Evidence Validator", "prioritization",
               "hypothesis-evidence-validator.md",
               "Keep weak or assumption-led experiments off the roadmap."),
    ModuleInfo("experiment_opportunity_scorer", "Experiment Opportunity Scorer", "prioritization",
               "experiment-opportunity-scorer.md",
               "Prioritize opportunities with the GROWTH framework."),
    ModuleInfo("growth_priority_orchestrator", "Growth Priority Orchestrator", "prioritization",
               "growth-priority-orchestrator.md",
               "Combine module outputs into a Now / Next / Later growth roadmap."),
    ModuleInfo("downstream_impact_analyzer", "Downstream Impact Analyzer", "learning",
               "downstream-impact-analyzer.md",
               "Check whether a win improved business quality, not just top-of-funnel conversion."),
    ModuleInfo("experiment_learning_capture", "Experiment Learning Capture", "learning",
               "experiment-learning-capture.md",
               "Turn every experiment, including losses, into reusable knowledge."),
    ModuleInfo("growth_council_insight_brief", "Growth Council Insight Brief", "learning",
               "growth-council-insight-brief.md",
               "Turn fragmented signals into a cross-functional decision brief."),
]
