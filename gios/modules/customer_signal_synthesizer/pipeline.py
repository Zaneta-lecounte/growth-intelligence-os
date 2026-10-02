"""Customer Signal Synthesizer pipeline: tag (LLM, batched) -> count (code) -> assess (LLM) -> rank (code)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from gios.core import data
from gios.core.llm import complete_json
from gios.modules.customer_signal_synthesizer import analysis
from gios.modules.customer_signal_synthesizer.models import (
    EvidenceTag,
    TagBatch,
    ThemeSynthesis,
)

TAG_PROMPT = "tag_customer_evidence"
SYNTH_PROMPT = "synthesize_customer_themes"


@dataclass
class CustomerSignalResult:
    tagged: pd.DataFrame
    themes: pd.DataFrame
    synthesis: ThemeSynthesis
    untagged: int
    disagreements: dict[str, int]
    meta: dict[str, Any] = field(default_factory=dict)


def tag_evidence(evidence: pd.DataFrame, client: Any = None) -> dict[str, EvidenceTag]:
    """Tag each distinct verbatim once, in batches."""
    unique = analysis.unique_verbatims(evidence)
    tags: dict[str, EvidenceTag] = {}
    for batch in analysis.batches(unique):
        result = complete_json(
            TAG_PROMPT, TagBatch,
            {"count": len(batch), "verbatims": analysis.batch_payload(batch)},
            client=client,
        )
        tags.update(analysis.merge_tags(result.tags, set(batch.key)))
    return tags


def synthesis_payload(themes: pd.DataFrame, tagged: pd.DataFrame, examples: int = 4) -> str:
    """Precomputed facts per theme for the assessment prompt (numbers are context only)."""
    items = []
    for r in themes.itertuples():
        g = tagged[tagged.theme == r.theme]
        items.append({
            "theme": r.theme,
            "verbatim_count": int(r.count),
            "share_of_evidence": f"{r.share * 100:.0f}%",
            "trend": r.trend,
            "segment": r.segment_text,
            "journey_stage": r.journey_stage_text,
            "top_sources": r.sources,
            "topics": sorted(g.topic.dropna().unique().tolist())[:8],
            "example_verbatims": g.verbatim.value_counts().index[:examples].tolist(),
        })
    return json.dumps(items, indent=2)


def assess_themes(themes: pd.DataFrame, tagged: pd.DataFrame, client: Any = None) -> ThemeSynthesis:
    return complete_json(SYNTH_PROMPT, ThemeSynthesis,
                         {"themes": synthesis_payload(themes, tagged)}, client=client)


def run(evidence: Optional[pd.DataFrame] = None, client: Any = None) -> CustomerSignalResult:
    evidence = data.load("customer_evidence") if evidence is None else evidence
    tags = tag_evidence(evidence, client)
    tagged = analysis.apply_tags(evidence, tags)
    counted = analysis.theme_table(tagged)
    synthesis = assess_themes(counted, tagged, client)
    themes = analysis.attach_scores(counted, synthesis.assessments)
    return CustomerSignalResult(
        tagged=tagged,
        themes=themes,
        synthesis=synthesis,
        untagged=int((~tagged.tagged).sum()),
        disagreements=analysis.disagreements(tagged),
        meta={"evidence_rows": len(evidence), "unique_verbatims": int(evidence.text_key.nunique()),
              "months": f"{evidence.month.min()} to {evidence.month.max()}"},
    )
