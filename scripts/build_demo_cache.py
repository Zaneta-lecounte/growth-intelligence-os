"""Build the DEMO MODE cache in demo/ for every LLM step.

    python scripts/build_demo_cache.py            # offline reference outputs (no API key needed)
    python scripts/build_demo_cache.py --live     # call the real model via gios.core.llm

Offline mode writes reference annotations authored by hand for the synthetic EchoAI data
(one annotation per verbatim template, written the way the tagging prompt asks). They are
used only as cached LLM outputs. Every file is validated against the step's Pydantic model.
--live re-runs each step against the API (ANTHROPIC_API_KEY must be set) and caches the result.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pydantic import BaseModel  # noqa: E402

from gios import config  # noqa: E402
from gios.core import data  # noqa: E402
from scripts import generate_data as gen  # noqa: E402

# ---------------------------------------------------------------------------------------------
# Customer Signal Synthesizer: reference tags per verbatim template
# (theme, topic, sentiment, segment, journey_stage, business_relevance)
# ---------------------------------------------------------------------------------------------

T = tuple[str, str, str, str, str, str]
CUSTOMER_TAGS: dict[str, T] = {
    # pricing pool
    gen.PRICING_CLARITY[0]: ("pricing_uncertainty", "plan inclusions unclear", "negative", "unknown", "evaluation", "high"),
    gen.PRICING_CLARITY[1]: ("pricing_uncertainty", "cannot estimate total cost", "negative", "unknown", "evaluation", "high"),
    gen.PRICING_CLARITY[2]: ("pricing_uncertainty", "usage caps unclear from paid search", "negative", "unknown", "evaluation", "high"),
    gen.PRICING_CLARITY[3]: ("pricing_uncertainty", "add-on versus included features", "negative", "unknown", "evaluation", "high"),
    gen.PRICING_CLARITY[4]: ("pricing_uncertainty", "tier differences unclear", "negative", "unknown", "purchase", "high"),
    gen.PRICING_CLARITY[5]: ("pricing_uncertainty", "pricing table fine print", "negative", "unknown", "evaluation", "high"),
    gen.PRICING_CLARITY[6]: ("pricing_uncertainty", "sales call needed for plan details", "negative", "unknown", "evaluation", "high"),
    gen.PRICING_CLARITY[7]: ("pricing_uncertainty", "pricing feels hidden", "negative", "smb", "evaluation", "high"),
    gen.PRICING_CLARITY[8]: ("pricing_uncertainty", "lost deal on plan confusion", "negative", "unknown", "purchase", "high"),
    gen.PRICING_CLARITY[9]: ("pricing_uncertainty", "overage billing unclear", "negative", "unknown", "purchase", "high"),
    gen.PRICING_CLARITY[10]: ("pricing_uncertainty", "plans indistinguishable after ad click", "negative", "unknown", "evaluation", "high"),
    # webinar pool
    gen.WEBINAR_MISMATCH[0]: ("urgency", "no budget, learning only", "neutral", "unknown", "awareness", "medium"),
    gen.WEBINAR_MISMATCH[1]: ("urgency", "student researcher, not a buyer", "positive", "unknown", "awareness", "low"),
    gen.WEBINAR_MISMATCH[2]: ("urgency", "no buying authority", "neutral", "mid_market", "consideration", "medium"),
    gen.WEBINAR_MISMATCH[3]: ("feature_fit", "team too small for sales-led demo", "neutral", "smb", "consideration", "medium"),
    gen.WEBINAR_MISMATCH[4]: ("urgency", "exploring with no timeline", "neutral", "unknown", "awareness", "medium"),
    gen.WEBINAR_MISMATCH[5]: ("switching_cost", "locked into existing contract", "neutral", "unknown", "consideration", "medium"),
    gen.WEBINAR_MISMATCH[6]: ("urgency", "attended for free template", "neutral", "unknown", "awareness", "low"),
    # mobile pool
    gen.MOBILE_FORM[0]: ("technical_issue", "mobile demo form hangs", "negative", "unknown", "evaluation", "high"),
    gen.MOBILE_FORM[1]: ("technical_issue", "mobile demo form freezes on submit", "negative", "unknown", "evaluation", "high"),
    gen.MOBILE_FORM[2]: ("technical_issue", "slow mobile demo page and date picker", "negative", "unknown", "evaluation", "high"),
    gen.MOBILE_FORM[3]: ("technical_issue", "abandoned mobile demo form", "negative", "unknown", "evaluation", "high"),
    gen.MOBILE_FORM[4]: ("technical_issue", "unresponsive submit button on mobile", "negative", "unknown", "evaluation", "high"),
    gen.MOBILE_FORM[5]: ("technical_issue", "mobile form loses input", "negative", "unknown", "evaluation", "high"),
    # red herring pool
    gen.RED_HERRING[0]: ("usability_polish", "dark mode request", "neutral", "unknown", "retention", "low"),
    gen.RED_HERRING[1]: ("usability_polish", "dark mode request", "neutral", "unknown", "retention", "low"),
    gen.RED_HERRING[2]: ("usability_polish", "export file naming", "negative", "unknown", "retention", "low"),
    gen.RED_HERRING[3]: ("usability_polish", "color themes", "neutral", "unknown", "retention", "low"),
    gen.RED_HERRING[4]: ("usability_polish", "dark mode request", "positive", "unknown", "retention", "low"),
    gen.RED_HERRING[5]: ("usability_polish", "transcript font choice", "neutral", "unknown", "retention", "low"),
    gen.RED_HERRING[6]: ("usability_polish", "emoji reactions", "neutral", "unknown", "retention", "low"),
    # noise pool
    gen.NOISE[0]: ("feature_fit", "accuracy with accents", "positive", "unknown", "retention", "medium"),
    gen.NOISE[1]: ("feature_fit", "native Teams support", "neutral", "unknown", "evaluation", "medium"),
    gen.NOISE[2]: ("trust_proof", "security certification required", "neutral", "mid_market", "purchase", "high"),
    gen.NOISE[3]: ("implementation_risk", "SSO setup unclear", "negative", "unknown", "onboarding", "medium"),
    gen.NOISE[4]: ("competitive_comparison", "better summaries than previous tool", "positive", "unknown", "purchase", "medium"),
    gen.NOISE[5]: ("technical_issue", "Zoom integration dropped recording", "negative", "unknown", "retention", "medium"),
    gen.NOISE[6]: ("feature_fit", "speaker labels with crosstalk", "negative", "unknown", "retention", "medium"),
    gen.NOISE[7]: ("feature_fit", "admin usage dashboard for expansion", "neutral", "unknown", "retention", "high"),
    gen.NOISE[8]: ("complexity", "helpful onboarding videos", "positive", "unknown", "onboarding", "low"),
    gen.NOISE[9]: ("trust_proof", "EU data residency", "neutral", "mid_market", "purchase", "high"),
    gen.NOISE[10]: ("competitive_comparison", "best action-item extraction", "positive", "unknown", "evaluation", "medium"),
    gen.NOISE[11]: ("technical_issue", "slow search in large workspaces", "negative", "unknown", "retention", "medium"),
}


def _template_regex(template: str) -> re.Pattern:
    escaped = re.escape(template)
    escaped = escaped.replace(re.escape("{plan}"), r"(?:.+?)").replace(re.escape("{phone}"), r"(?:.+?)")
    return re.compile(f"^{escaped}$")


_TEMPLATE_PATTERNS = [(_template_regex(t), tag) for t, tag in CUSTOMER_TAGS.items()]


def reference_tag(verbatim: str) -> T:
    for pattern, tag in _TEMPLATE_PATTERNS:
        if pattern.match(verbatim):
            return tag
    raise KeyError(f"no reference annotation for verbatim: {verbatim!r}")


THEME_REFERENCE = {
    # theme: (severity, commercial, journey, summary, expected_behavior, funnel_stage, analytics_signal, question)
    "pricing_uncertainty": (5, 5, 5,
        "Evaluators cannot tell what each plan includes or what they would pay, and some deals stall or are lost on it.",
        "High exits and low demo-CTA clicks on the pricing page, repeat pricing visits, pricing questions in chat.",
        "Evaluation, from pricing page to demo request",
        "Pricing-page exit rate and CTA rate by source, especially paid search; pricing page revisits per session.",
        "Does clearer package framing on the pricing page improve qualified demo requests without reducing lead quality?"),
    "technical_issue": (4, 4, 4,
        "Prospects hit broken or slow experiences, most acutely on the mobile demo form; customers also report integration and search defects.",
        "Mobile form starts without completion, rage clicks, slow load on the demo page.",
        "Evaluation, at demo form submission",
        "Mobile versus desktop demo-form completion, rage clicks, and page load time.",
        "Does fixing mobile demo-form performance increase completed demo requests without lowering lead quality?"),
    "trust_proof": (4, 4, 4,
        "Larger buyers need security and data-residency proof before they can sign.",
        "Late-stage deals stall; security-page visits from opportunity accounts.",
        "Purchase, during security and legal review",
        "Opportunity-to-win rate and cycle time for mid-market deals that request security documents.",
        "Does a self-serve trust center shorten mid-market deal cycles without adding sales effort?"),
    "urgency": (3, 4, 3,
        "Many webinar-sourced contacts have no budget, authority, or active project; they are learning, not buying.",
        "Strong lead-to-MQL conversion but weak MQL-to-SQL acceptance for webinar leads.",
        "Consideration, at the MQL to SQL handoff",
        "Sales rejection reasons and MQL-to-SQL rate by source.",
        "Does adding buying-intent questions to webinar registration improve MQL-to-SQL conversion without cutting qualified attendance?"),
    "feature_fit": (3, 3, 2,
        "Requests for integrations, accuracy, and admin features that shape expansion more than acquisition.",
        "Feature-page engagement and expansion conversations.",
        "Retention and expansion",
        "Feature usage and expansion rate for accounts requesting integrations or admin tools.",
        "Does an admin usage dashboard increase seat expansion without raising support load?"),
    "competitive_comparison": (2, 3, 3,
        "Buyers compare tools and tend to choose EchoAI on summary and action-item quality.",
        "Compare-page visits before conversion.",
        "Evaluation",
        "Compare-page engagement and its conversion rate.",
        "Does leading with summary quality on the compare page improve win rate without slowing evaluation?"),
    "implementation_risk": (3, 2, 2,
        "Setup steps such as SSO are unclear and slow onboarding.",
        "Long time-to-first-meeting for SSO accounts.",
        "Onboarding",
        "Time to first transcribed meeting by SSO status.",
        "Does a guided SSO setup shorten time-to-value without increasing support tickets?"),
    "switching_cost": (2, 2, 2,
        "Some prospects are tied to existing contracts.",
        "Nurture engagement without near-term conversion.",
        "Consideration",
        "Lead age at conversion for contract-locked prospects.",
        "Does a contract-renewal nurture track raise later conversion without increasing unsubscribes?"),
    "complexity": (1, 1, 1,
        "Onboarding content is seen as helpful; little evidence of complexity problems.",
        "Healthy activation after onboarding videos.",
        "Onboarding",
        "Activation rate for accounts that watch onboarding videos.",
        "Does surfacing onboarding videos earlier improve activation without adding friction?"),
    "usability_polish": (1, 1, 1,
        "Frequent but low-stakes cosmetic requests such as dark mode and file naming, raised by existing customers.",
        "None expected in acquisition metrics.",
        "Retention, cosmetic",
        "Satisfaction scores among customers requesting cosmetic changes.",
        "Does shipping dark mode change retention or satisfaction without diverting roadmap capacity?"),
}

THEME_TENSIONS = [
    "The most frequent request, cosmetic polish such as dark mode, comes from happy customers and does not block buying; the costliest problem, pricing clarity, is less frequent but blocks evaluators.",
    "Webinars attract curious learners who rate the content highly, but those same contacts rarely have budget or authority to buy.",
    "Evaluators want to self-serve pricing and demo booking, yet unclear plans and a broken mobile form push them into sales calls or away entirely.",
    "Mid-market buyers praise product quality but cannot sign until security and data-residency proof is available.",
]
THEME_GAPS = [
    "We do not know which specific plan elements cause pricing confusion; run short pricing-page intercept surveys.",
    "Evidence is not linked to lead or account records, so we cannot tie themes to pipeline value directly.",
    "Mobile form complaints come from chat and support only; session recordings would confirm where the form fails.",
    "Win and loss notes are sparse for mid-market deals, so the weight of trust and proof issues is uncertain.",
]


def build_customer_signal(live: bool) -> None:
    from gios.modules.customer_signal_synthesizer import analysis, pipeline
    from gios.modules.customer_signal_synthesizer.models import (
        EvidenceTag,
        TagBatch,
        ThemeAssessment,
        ThemeSynthesis,
    )

    evidence = data.load("customer_evidence")
    if live:
        tags = pipeline.tag_evidence(evidence)
    else:
        unique = analysis.unique_verbatims(evidence)
        tags = {}
        for r in unique.itertuples():
            theme, topic, sentiment, segment, stage, relevance = reference_tag(r.verbatim)
            tags[r.key] = EvidenceTag(key=r.key, theme=theme, topic=topic, sentiment=sentiment,
                                      segment=segment, journey_stage=stage, business_relevance=relevance)
    _write(pipeline.TAG_PROMPT, TagBatch(tags=list(tags.values())))

    tagged = analysis.apply_tags(evidence, tags)
    themes = analysis.theme_table(tagged)
    if live:
        synthesis = pipeline.assess_themes(themes, tagged)
    else:
        assessments = []
        for theme in themes.theme:
            s, c, j, summary, behavior, stage, signal, question = THEME_REFERENCE[theme]
            assessments.append(ThemeAssessment(
                theme=theme, severity=s, commercial_relevance=c, journey_relevance=j, summary=summary,
                expected_behavior=behavior, funnel_stage=stage, analytics_signal=signal,
                testable_question=question))
        synthesis = ThemeSynthesis(assessments=assessments, tensions=THEME_TENSIONS, research_gaps=THEME_GAPS)
    _write(pipeline.SYNTH_PROMPT, synthesis)


# ---------------------------------------------------------------------------------------------


def _write(name: str, obj: BaseModel) -> None:
    path = config.DEMO_DIR / f"{name}.json"
    payload: Any = json.loads(obj.model_dump_json())
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    type(obj).model_validate_json(path.read_text())
    print(f"wrote {path.relative_to(ROOT)}")


BUILDERS: dict[str, Callable[[bool], None]] = {
    "customer_signal": build_customer_signal,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build demo/ cached LLM outputs.")
    parser.add_argument("--live", action="store_true", help="call the real model (needs ANTHROPIC_API_KEY)")
    parser.add_argument("--only", choices=sorted(BUILDERS), help="build one module only")
    args = parser.parse_args()
    if args.live and config.is_demo_mode():
        parser.error("--live needs ANTHROPIC_API_KEY")
    for name, builder in BUILDERS.items():
        if args.only in (None, name):
            builder(args.live)


if __name__ == "__main__":
    main()
