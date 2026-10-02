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
    deterministic: str = ""
    llm: str = ""

    @property
    def spec_path(self):
        return config.SPECS_DIR / self.spec


MODULES: list[ModuleInfo] = [
    ModuleInfo("customer_signal_synthesizer", "Customer Signal Synthesizer", "signal",
               "customer-signal-synthesizer.md",
               "Turn qualitative customer evidence into structured themes without overstating frequency.",
               status="ready",
               deterministic='frequency counted from tags and binned 1–5, trend, dominant segment and stage, overall score F×S×C×J and ranking, signals.',
               llm='tags each distinct verbatim (theme, sentiment, segment, stage, relevance) in batches; proposes severity, commercial and journey relevance (editable); writes tensions, gaps and testable questions.'),
    ModuleInfo("behavioral_friction_analyzer", "Behavioral Friction Analyzer", "signal",
               "behavioral-friction-analyzer.md",
               "Find where digital behavior signals confusion, hesitation, or abandonment.",
               status="ready",
               deterministic='anomaly detection (mix-standardized device and source divergence, load time vs site baseline, month trend) with z and deviation thresholds; funnel sizing; the customer-evidence check on causes.',
               llm='classifies friction type and writes competing explanations, validation needed and the recommended action; may only cite customer signals that exist.'),
    ModuleInfo("growth_intelligence_diagnostic", "Growth Intelligence Diagnostic", "diagnosis",
               "growth-intelligence-diagnostic.md",
               "Identify the most important growth problem and the next best action.",
               status="ready",
               deterministic='the Step 1 gate, data-drafted baselines, signal filtering and classification, the leakage point (from the Leakage Auditor), hypothesis ranking and confidence.',
               llm='writes two to five root-cause hypotheses with supporting, contradictory and missing evidence, the next best action and the measurement plan; every claim must cite an in-scope signal id.'),
    ModuleInfo("qualified_demand_leakage_auditor", "Qualified Demand Leakage Auditor", "diagnosis",
               "qualified-demand-leakage-auditor.md",
               "Find where qualified demand is lost between acquisition and revenue.",
               status="ready",
               deterministic='everything numeric: stage map with Wilson 95% CIs, peer benchmarks, minimum volume, all leakage flags, gap-closure sizing, owner classification.',
               llm='writes the narrative summary, cause explanations and intervention only, pinned to the leak it describes.'),
    ModuleInfo("hypothesis_evidence_validator", "Hypothesis Evidence Validator", "diagnosis",
               "hypothesis-evidence-validator.md",
               "Keep weak or assumption-led experiments off the roadmap.",
               status="ready",
               deterministic='the six-part standard check, evidence map, evidence caps on each score, total, band, recommendation and weakest area.',
               llm='proposes each 0–2 score with a one-line justification (capped by code, overridable by you) and lists missing evidence.'),
    ModuleInfo("experiment_opportunity_scorer", "Experiment Opportunity Scorer", "prioritization",
               "experiment-opportunity-scorer.md",
               "Prioritize opportunities with the GROWTH framework.",
               status="ready",
               deterministic='R, O, W prefills from linked signals, H from the validator total, the GROWTH priority score, the sample-size check, recommendation rules.',
               llm='none. G and T are your judgment; every default is editable.'),
    ModuleInfo("growth_priority_orchestrator", "Growth Priority Orchestrator", "prioritization",
               "growth-priority-orchestrator.md",
               "Combine module outputs into a Now / Next / Later growth roadmap.",
               status="ready",
               deterministic='merging confirmed duplicates, portfolio buckets, minimum dependencies, default horizons and ordering, the portfolio balance check.',
               llm='proposes duplicate clusters (you confirm) and tags dependencies, expected learning and the primary metric per item.'),
    ModuleInfo("downstream_impact_analyzer", "Downstream Impact Analyzer", "learning",
               "downstream-impact-analyzer.md",
               "Check whether a win improved business quality, not just top-of-funnel conversion.",
               status="ready",
               deterministic='everything: two-proportion tests with 95% CIs at every stage, quality rates, revenue per visitor, CAC, power, observation window vs velocity, segment interactions, interpretation and recommendation rules.',
               llm='none.'),
    ModuleInfo("experiment_learning_capture", "Experiment Learning Capture", "learning",
               "experiment-learning-capture.md",
               "Turn every experiment, including losses, into reusable knowledge.",
               status="ready",
               deterministic="the statistical result, segment findings, default decision, library search and filters; saving is blocked without 'What we should NOT conclude'.",
               llm='drafts each narrative section, a theme tag and a six-part next hypothesis; you edit before saving.'),
    ModuleInfo("growth_council_insight_brief", "Growth Council Insight Brief", "learning",
               "growth-council-insight-brief.md",
               "Turn fragmented signals into a cross-functional decision brief.",
               status="ready",
               deterministic="gathering the period's records, the numbers shown next to each claim, owner suggestions from the Leakage Auditor owner classes, the eight-line Slack summary.",
               llm='writes the ten sections along Signal → Why it matters → Evidence → Hypothesis → Action → Owner → Learning, citing only records in scope; picks an owner class (editable).'),
]
