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
    gen.PRICING_CLARITY[11]: ("pricing_uncertainty", "unsure which plan fits team size", "negative", "unknown", "evaluation", "high"),
    gen.PRICING_CLARITY[12]: ("pricing_uncertainty", "sales contact needed for a price", "negative", "unknown", "consideration", "high"),
    gen.PRICING_CLARITY[13]: ("pricing_uncertainty", "demo required before seeing cost", "negative", "unknown", "evaluation", "high"),
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
# Behavioral Friction Analyzer: reference classifications for default-threshold findings
# ---------------------------------------------------------------------------------------------

FRICTION_REFERENCE = {
    "demo|device=mobile": dict(
        friction_type="technical",
        what_happened=("Mobile visitors start the demo form at least as often as desktop visitors but "
                       "complete it far less often, leave the page more often, rage-click much more, and wait much "
                       "longer for the page to load."),
        competing_explanations=[
            "A technical defect on mobile, such as a slow or blocking script, breaks submission or the date picker.",
            "Mobile visitors are earlier-stage browsers who start the form casually and intend to finish on a laptop.",
            "Form completion events fail to fire on mobile, so completions are under-counted rather than lost.",
        ],
        validation_needed=[
            "Session recordings of mobile demo-form attempts to see where users stall.",
            "Whether mobile starters later complete the form on desktop (cross-device lead matching).",
            "A server-side count of demo requests by device to rule out a tracking gap.",
            "Customer chat and support verbatims mentioning the mobile demo form.",
        ],
        supporting_customer_signal_ids=["css-technical_issue"],
        recommended_action="technical_fix",
        recommended_detail=("Profile and fix mobile demo-page performance and the submit and date-picker "
                            "interactions, verify completion tracking, then confirm the recovery with a "
                            "before and after comparison of mobile completion."),
    ),
    "pricing|source=paid_search": dict(
        friction_type="comprehension",
        what_happened=("Paid search visitors leave the pricing page far more often than other traffic and "
                       "click through to a demo less often, and the gap has widened every month."),
        competing_explanations=[
            "Plans and inclusions are hard to compare, so evaluators cannot judge fit or cost.",
            "Expectation mismatch: ad copy or keywords promise something the pricing page does not show.",
            "Traffic quality shifted as bids rose, bringing in more price-shoppers with lower intent.",
        ],
        validation_needed=[
            "Customer verbatims about pricing clarity, especially from paid search visitors.",
            "Search query and ad copy review for the keywords landing on pricing.",
            "A pricing-page intercept survey asking what is missing or unclear.",
        ],
        supporting_customer_signal_ids=["css-pricing_uncertainty"],
        recommended_action="ux_experiment",
        recommended_detail=("Test a clearer plan comparison with explicit inclusions and usage limits for "
                            "paid search traffic, measured on demo requests and downstream SQL rate."),
    ),
    "compare|device=desktop": dict(
        friction_type="navigation",
        what_happened=("Desktop visitors on the compare page rage-click slightly more than mobile visitors; "
                       "the absolute difference is small."),
        competing_explanations=[
            "Chance variation: rage clicks are rare, and mobile is unusually low here rather than desktop being high.",
            "A desktop-only interactive element on the compare page, such as a table toggle, that looks clickable but is not.",
        ],
        validation_needed=[
            "Whether the gap persists in the next month of data.",
            "Click maps of the compare page on desktop.",
        ],
        supporting_customer_signal_ids=[],
        recommended_action="instrumentation_improvement",
        recommended_detail="Monitor rather than act: add element-level click tracking on the compare table and re-check next month.",
    ),
}


def build_behavioral_friction(live: bool) -> None:
    from gios.core.schemas import Signal
    from gios.modules.behavioral_friction_analyzer import analysis, pipeline
    from gios.modules.behavioral_friction_analyzer.models import FrictionBatch, FrictionClassification

    web, funnel = data.load("web_behavior"), data.load("funnel_by_source")
    findings = analysis.detect(web)
    analysis.size_all(findings, web, funnel)
    if live:
        from gios.modules.customer_signal_synthesizer import analysis as css_analysis
        from gios.modules.customer_signal_synthesizer import run as css_run

        customer: list[Signal] = css_analysis.to_signals(css_run().themes)
        classifications = pipeline.classify(findings, customer)
        batch = FrictionBatch(classifications=list(classifications.values()))
    else:
        missing = {f.id for f in findings} - set(FRICTION_REFERENCE)
        if missing:
            raise SystemExit(f"no reference classification for findings: {sorted(missing)}")
        batch = FrictionBatch(classifications=[
            FrictionClassification(finding_id=f.id, **FRICTION_REFERENCE[f.id]) for f in findings])
    _write(pipeline.CLASSIFY_PROMPT, batch)


# ---------------------------------------------------------------------------------------------
# Qualified Demand Leakage Auditor: reference narrative for the default settings
# ---------------------------------------------------------------------------------------------

LEAKAGE_REFERENCE = dict(
    leakage_summary=(
        "Webinars are EchoAI's largest lead source and pass leads to MQL more readily than any other "
        "channel, but sales accepts very few of those MQLs. The loss happens at the MQL to SQL handoff; once "
        "webinar leads are accepted they progress like any other source. Sales rejection reasons point to a "
        "qualification mismatch: low intent, purely educational interest, or no active buying need. Follow-up "
        "speed for webinar leads is in line with other sources, so this is not an SLA problem. Separately, partner leads wait longer for first contact and a small share of MQLs is "
        "never routed."),
    probable_causes=[
        dict(owner="qualification", explanation=(
            "The MQL logic counts webinar attendance and content engagement as buying intent, so learners and "
            "researchers become MQLs and sales then rejects them.")),
        dict(owner="acquisition", explanation=(
            "Webinar topics and promotion attract a broad, educational audience rather than active evaluators, so "
            "volume rises without buying intent.")),
    ],
    recommended_intervention=(
        "Add buying-intent questions such as role, team size, and project timeline to webinar registration, "
        "and require an intent signal before a webinar attendee becomes an MQL. Route non-intent attendees to "
        "nurture instead of sales. Guardrail: qualified webinar pipeline and attendance must not fall."),
    primary_metric="Webinar MQL to SQL conversion, measured by sales acceptance of webinar-sourced MQLs.",
    downstream_metrics=[
        "Webinar SQL to opportunity and opportunity to win conversion, to confirm accepted leads still close.",
        "Webinar-sourced pipeline and revenue, so tightening qualification does not shrink real demand.",
        "Sales hours spent per accepted webinar SQL.",
    ],
)


def build_demand_leakage(live: bool) -> None:
    from gios.modules.qualified_demand_leakage_auditor import analysis, pipeline
    from gios.modules.qualified_demand_leakage_auditor.models import LeakageNarrative

    a = analysis.analyze(data.load("funnel_by_source"), data.load("sales_feedback"))
    if live:
        narrative, note = pipeline.narrate(a)
        if narrative is None:
            raise SystemExit(note)
    else:
        narrative = LeakageNarrative(about_leak=pipeline.leak_id(a.top_leak), **LEAKAGE_REFERENCE)
        if narrative.about_leak != "webinar|MQL→SQL":
            raise SystemExit(f"reference narrative is about webinar MQL→SQL, data says {narrative.about_leak}")
    _write(pipeline.NARRATE_PROMPT, narrative)


# ---------------------------------------------------------------------------------------------
# Growth Intelligence Diagnostic: reference diagnoses per channel scope
# ---------------------------------------------------------------------------------------------


def _claim(signal_id: str, claim: str, status: str = "observed") -> dict:
    return {"signal_id": signal_id, "claim": claim, "status": status}


PRICING_HYPOTHESIS = dict(
    title="Unclear pricing is losing paid search evaluators",
    observed_problem=("Paid search visitors exit the pricing page far more often than other traffic, fewer of "
                      "them click through to a demo, and paid search acquisition cost has climbed."),
    affected_audience="Paid search evaluators, mostly small businesses, at the pricing step",
    causal_explanation=("Evaluators cannot tell what each plan includes or what they would pay, so high-intent "
                        "search visitors leave instead of requesting a demo."),
    intervention="A clearer plan comparison with explicit inclusions, usage limits, and an estimate of total cost.",
    expected_behavior_change="Lower pricing-page exit rate and higher demo click-through for paid search traffic.",
    expected_business_outcome="More qualified demo requests per paid search dollar, bringing acquisition cost back down.",
    supporting=[
        _claim("css-pricing_uncertainty", "Customers repeatedly say they cannot tell what plans include, and these "
               "comments are rising; one names arriving from a search ad."),
        _claim("bfa-pricing-source-paid_search", "Paid search exits on the pricing page are well above other "
               "sources and rising each month, with fewer demo clicks."),
        _claim("qdl-flag-rising_cac-paid_search", "Paid search acquisition cost rose sharply as spend grew while "
               "wins fell."),
        _claim("qdl-leak-paid_search-visit-to-lead", "Fewer paid search visits become leads than benchmark, "
               "consistent with evaluators leaving before converting.", "inferred"),
    ],
    contradicting=[],
    missing=["Pricing verbatims are not tagged by acquisition channel, so the paid search link rests on a few mentions.",
             "No pricing-page survey or session recordings show which plan details confuse visitors."],
)
MOBILE_HYPOTHESIS = dict(
    title="A broken mobile demo form loses ready-to-buy visitors",
    observed_problem="Mobile visitors start the demo form often but rarely finish it, rage-click, and wait on a slow page.",
    affected_audience="Mobile visitors on the demo page across channels",
    causal_explanation="A technical defect or slow script on the mobile demo page blocks form submission.",
    intervention="Profile and fix the mobile demo page and form, then verify completion tracking.",
    expected_behavior_change="Mobile demo-form completion rises toward the desktop rate.",
    expected_business_outcome="More demo requests and pipeline without additional spend.",
    supporting=[
        _claim("bfa-demo-device-mobile", "Mobile demo-form completion is far below desktop, with many more rage "
               "clicks and much slower loads."),
        _claim("css-technical_issue", "Prospects report the demo form spinning or freezing on their phones."),
    ],
    contradicting=[],
    missing=["We cannot yet tell how many mobile starters later complete the form on desktop.",
             "Server-side demo request counts by device would rule out a tracking gap."],
)

DIAGNOSTIC_REFERENCE = {
    "paid_search": dict(
        growth_problem=("Paid search is getting more expensive while converting fewer evaluators. The loss "
                        "concentrates on the pricing page, where paid search visitors increasingly leave without "
                        "requesting a demo."),
        hypotheses=[
            PRICING_HYPOTHESIS,
            dict(
                title="Rising bids are buying lower-intent paid search traffic",
                observed_problem="Paid search spend rose while fewer of its leads qualify and fewer deals close.",
                affected_audience="Paid search visitors across the site",
                causal_explanation="Higher bids and broader keywords bring in price-shoppers with weaker intent.",
                intervention="Tighten keyword targeting and negative keywords for low-intent queries.",
                expected_behavior_change="Higher lead-to-MQL conversion for paid search at similar volume.",
                expected_business_outcome="Lower acquisition cost through better traffic quality.",
                supporting=[
                    _claim("qdl-flag-rising_cac-paid_search", "Spend grew while wins fell, so each win costs more."),
                    _claim("qdl-leak-paid_search-lead-to-mql", "Paid search leads qualify less often than "
                           "benchmark, which fits weaker intent.", "inferred"),
                ],
                contradicting=[
                    _claim("bfa-pricing-source-paid_search", "The drop concentrates on the pricing page and in "
                           "demo clicks there, which a site-wide traffic-quality shift would not predict.",
                           "inferred"),
                ],
                missing=["Search query and keyword mix reports over the period.",
                         "Lead quality by keyword group."],
            ),
            MOBILE_HYPOTHESIS,
        ],
        next_best_action="experiment",
        action_detail=("Run a pricing-page experiment for paid search traffic: a clear plan comparison with "
                       "inclusions and usage limits against the current page, while a short intercept survey "
                       "captures what visitors find unclear."),
        primary_metric="Demo requests per paid search pricing-page session.",
        guardrail_metric="Lead-to-SQL rate for paid search, so clearer pricing does not attract unqualified leads.",
        downstream_metric="Paid search acquisition cost and wins over the following quarter.",
        learning_objective="Learn whether pricing clarity, rather than traffic quality, drives the paid search decline.",
        missing_evidence=["Channel-level tagging of customer evidence.",
                          "Paid search keyword and query mix over time."],
    ),
    "all": dict(
        growth_problem=("Qualified pipeline is leaking most at the webinar handoff: webinars bring the most leads "
                        "but sales accepts very few of them. Paid search economics and the mobile demo form are "
                        "smaller, separate leaks."),
        hypotheses=[
            dict(
                title="Webinar leads are learners, not buyers",
                observed_problem="Webinar leads become MQLs readily but sales rejects most of them.",
                affected_audience="Webinar attendees, mostly small businesses",
                causal_explanation=("The MQL rules treat webinar engagement as buying intent, so learners and "
                                    "researchers with no active project reach sales."),
                intervention="Add buying-intent questions to registration and require an intent signal before MQL.",
                expected_behavior_change="Fewer but better webinar MQLs; sales accepts a much larger share.",
                expected_business_outcome="More SQLs per sales hour and steady webinar-sourced pipeline.",
                supporting=[
                    _claim("qdl-leak-webinar-mql-to-sql", "Webinar MQL to SQL conversion is far below other sources "
                           "on large volume."),
                    _claim("qdl-flag-qualification_mismatch-webinar", "Most webinar rejections cite low intent, "
                           "educational interest only, or not being in market."),
                    _claim("qdl-flag-strong_top_weak_downstream-webinar", "Webinars convert leads to MQL better "
                           "than any source but fail downstream."),
                    _claim("qdl-flag-high_volume_low_quality-webinar", "Webinars are the largest lead source with "
                           "the lowest lead-to-win rate."),
                    _claim("css-urgency", "Webinar attendees describe learning or research with no project or "
                           "budget.", "inferred"),
                ],
                contradicting=[],
                missing=["How the MQL score weights webinar attendance.",
                         "Whether rejected webinar contacts convert later through nurture."],
            ),
            PRICING_HYPOTHESIS,
            MOBILE_HYPOTHESIS,
            dict(
                title="Slow partner follow-up loses partner demand",
                observed_problem="Many partner leads wait longer than a day for first contact.",
                affected_audience="Partner-referred leads",
                causal_explanation="Partner leads route to a separate team that responds slowly.",
                intervention="Route partner leads through the same response-time rules as direct leads.",
                expected_behavior_change="Faster first contact for partner leads.",
                expected_business_outcome="Higher partner acceptance and win rates.",
                supporting=[
                    _claim("qdl-flag-sla_loss-partner", "A large share of partner MQLs is contacted after the SLA."),
                ],
                contradicting=[
                    _claim("qdl-flag-sla_loss-partner", "Late partner leads are accepted about as often as "
                           "on-time ones, so delay has no measurable cost yet.", "inferred"),
                ],
                missing=["Partner win rates by response time."],
            ),
        ],
        next_best_action="process_change",
        action_detail=("Change the webinar qualification process: add intent questions to registration, require an "
                       "intent signal before a webinar contact becomes an MQL, and route the rest to nurture."),
        primary_metric="Webinar MQL to SQL conversion.",
        guardrail_metric="Webinar-sourced qualified pipeline must not fall.",
        downstream_metric="Webinar SQL to opportunity, win rate, and revenue.",
        learning_objective="Learn whether intent screening raises webinar lead quality without shrinking real demand.",
        missing_evidence=["MQL scoring rules and their weights.",
                          "Lead-level linkage between customer evidence and funnel records."],
    ),
}

from gios.modules.demo_flow import CACHED_DIAGNOSTIC_SCOPES, diagnostic_inputs  # noqa: E402


DIAGNOSTIC_REFERENCE["webinar"] = dict(
    growth_problem=("Webinars produce the most leads and the strongest lead-to-MQL conversion, but most webinar "
                    "MQLs never become sales-qualified, so webinar volume is not turning into pipeline."),
    hypotheses=[
        DIAGNOSTIC_REFERENCE["all"]["hypotheses"][0],
        dict(
            title="Webinar topics attract an educational audience",
            observed_problem="Webinars draw the largest lead volume but the lowest lead-to-win rate of any source.",
            affected_audience="Webinar registrants",
            causal_explanation="Webinar topics and promotion target learners rather than teams with an active buying need.",
            intervention="Shift webinar topics toward evaluation questions such as rollout, security and pricing.",
            expected_behavior_change="A larger share of registrants describe an active project.",
            expected_business_outcome="Higher webinar lead-to-win rate at similar cost.",
            supporting=[
                _claim("qdl-flag-high_volume_low_quality-webinar", "Webinars are the largest lead source with "
                       "the lowest lead-to-win rate."),
                _claim("css-urgency", "Attendees describe learning or research with no project or budget.",
                       "inferred"),
            ],
            contradicting=[],
            missing=["Registrant roles and company sizes by webinar topic.",
                     "Whether evaluation-focused webinars draw fewer registrants."],
        ),
    ],
    next_best_action="process_change",
    action_detail=("Add intent questions to webinar registration and require an intent signal before MQL; route "
                   "the rest to nurture."),
    primary_metric="Webinar MQL to SQL conversion.",
    guardrail_metric="Webinar-sourced qualified pipeline must not fall.",
    downstream_metric="Webinar SQL to opportunity and win rate.",
    learning_objective="Learn whether intent screening fixes webinar lead quality without shrinking real demand.",
    missing_evidence=["MQL scoring rules and their weights."],
)




def _phase1_signals():
    from gios.core.store import Store
    from gios.modules.signal_layer import run_signal_layer
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp) / "seed.db")
        run_signal_layer(store)
        from gios.core.schemas import Signal

        return store.list(Signal)


def build_growth_diagnostic(live: bool) -> None:
    from gios.modules.growth_intelligence_diagnostic import analysis as gid
    from gios.modules.growth_intelligence_diagnostic import pipeline
    from gios.modules.growth_intelligence_diagnostic.models import checked_output_model

    signals = _phase1_signals()
    for scope in CACHED_DIAGNOSTIC_SCOPES:
        business, filters = diagnostic_inputs(scope)
        if live:
            output = pipeline.run(business, filters, signals).output
        else:
            in_scope = gid.filter_signals(signals, filters.channel, filters.segment, filters.start, filters.end)
            output = checked_output_model(frozenset(s.id for s in in_scope))(**DIAGNOSTIC_REFERENCE[scope])
        _write(f"{pipeline.PROMPT}__{scope}", output)


# ---------------------------------------------------------------------------------------------
# Hypothesis Evidence Validator: reference score proposals per hypothesis
# ---------------------------------------------------------------------------------------------

VALIDATOR_REFERENCE = {
    # title: ({dimension: (score, justification)}, missing_evidence)
    "Unclear pricing is losing paid search evaluators": ({
        "evidence_diversity": (2, "Customer, behavioral, funnel and acquisition signals point the same way."),
        "behavioral_support": (2, "Pricing-page exits and demo clicks for paid search directly show the behavior."),
        "customer_support": (2, "Pricing-clarity comments recur and are rising."),
        "business_relevance": (2, "Directly tied to paid search acquisition cost and demo requests."),
        "testability": (2, "A pricing-page variant for paid search traffic is a clean causal test."),
        "measurement_readiness": (1, "Exit and click rates are tracked, but customer comments are not tied to channel."),
    }, ["Which plan details confuse visitors, from an intercept survey or recordings."]),
    "A broken mobile demo form loses ready-to-buy visitors": ({
        "evidence_diversity": (2, "Behavioral data and customer reports agree."),
        "behavioral_support": (2, "Mobile completion, rage clicks and load time directly show the friction."),
        "customer_support": (2, "Prospects repeatedly describe the form failing on their phones."),
        "business_relevance": (1, "It affects demo requests broadly rather than the target metric itself."),
        "testability": (2, "A before and after comparison of a fix isolates the effect."),
        "measurement_readiness": (1, "Mobile completion events may not fire reliably; tracking needs verifying."),
    }, ["Server-side demo request counts by device."]),
    "Rising bids are buying lower-intent paid search traffic": ({
        "evidence_diversity": (1, "Both signals come from the same funnel data, so they are not independent."),
        "behavioral_support": (0, "No behavioral signal supports it, and pricing-page behavior points elsewhere."),
        "customer_support": (0, "No customer evidence speaks to traffic quality."),
        "business_relevance": (2, "Directly tied to paid search acquisition cost."),
        "testability": (1, "Keyword changes are confounded with seasonality and bid shifts."),
        "measurement_readiness": (1, "No keyword-level lead quality reporting exists yet."),
    }, ["Search query and keyword mix over the period.", "Lead quality by keyword group."]),
    "Webinar leads are learners, not buyers": ({
        "evidence_diversity": (2, "Funnel, acquisition and customer signals agree."),
        "behavioral_support": (0, "No web behavior is linked to this hypothesis."),
        "customer_support": (1, "Attendee comments fit, but their link to webinars is inferred."),
        "business_relevance": (2, "Directly tied to MQL to SQL conversion, the target metric."),
        "testability": (2, "Intent questions at registration can be tested against current rules."),
        "measurement_readiness": (2, "Sales acceptance and rejection reasons are recorded for every MQL."),
    }, ["How the MQL score weights webinar attendance."]),
    "Slow partner follow-up loses partner demand": ({
        "evidence_diversity": (1, "A single operational signal."),
        "behavioral_support": (0, "No behavioral evidence."),
        "customer_support": (0, "No customer evidence."),
        "business_relevance": (1, "Partner volume is small and late leads are accepted at a similar rate."),
        "testability": (2, "Routing partner leads under the standard SLA is a clean test."),
        "measurement_readiness": (2, "Follow-up times and acceptance are recorded."),
    }, ["Partner win rates by response time."]),
    "Homepage chatbot will lift demo requests": ({
        "evidence_diversity": (0, "No evidence is linked; the hypothesis rests on assumption."),
        "behavioral_support": (0, "No behavioral signal shows visitors stalling on unanswered questions."),
        "customer_support": (0, "No customer evidence asks for live help on the homepage."),
        "business_relevance": (2, "Demo requests are the target outcome."),
        "testability": (2, "A chatbot on and off test is a clean causal design."),
        "measurement_readiness": (2, "Demo requests are already tracked."),
    }, ["Any customer or behavioral evidence that visitors leave because questions go unanswered."]),
}


VALIDATOR_REFERENCE["Webinar topics attract an educational audience"] = ({
    "evidence_diversity": (2, "Acquisition and customer signals point the same way."),
    "behavioral_support": (0, "No behavioral evidence is linked."),
    "customer_support": (1, "Attendee comments fit, but their link to webinar topics is inferred."),
    "business_relevance": (2, "Directly tied to webinar lead quality and pipeline."),
    "testability": (1, "Topic changes are confounded with seasonality and promotion."),
    "measurement_readiness": (1, "Registrant role and company size are not captured consistently."),
}, ["Registrant roles and company sizes by webinar topic."])


def build_hypothesis_validator(live: bool) -> None:
    from gios.modules.growth_intelligence_diagnostic import pipeline as gid
    from gios.modules.hypothesis_evidence_validator import analysis, pipeline
    from gios.modules.hypothesis_evidence_validator.examples import ASSUMPTION_ONLY
    from gios.modules.hypothesis_evidence_validator.models import DimensionScore, ValidatorProposal

    signals = _phase1_signals()
    hypotheses = {ASSUMPTION_ONLY.fingerprint(): ASSUMPTION_ONLY}
    for scope in CACHED_DIAGNOSTIC_SCOPES:
        business, filters = diagnostic_inputs(scope)
        for h in gid.to_hypotheses(gid.run(business, filters, signals)):
            hypotheses.setdefault(h.fingerprint(), h)
    for fp, h in hypotheses.items():
        if live:
            proposal, error = pipeline.propose(h, analysis.evidence_map(h, signals), analysis.check_standard(h))
            if proposal is None:
                raise SystemExit(error)
        else:
            scores, missing = VALIDATOR_REFERENCE[h.title]
            proposal = ValidatorProposal(
                scores=[DimensionScore(dimension=d, score=sc, justification=j) for d, (sc, j) in scores.items()],
                missing_evidence=missing)
        _write(f"{pipeline.PROMPT}__{fp}", proposal)


# ---------------------------------------------------------------------------------------------
# Growth Priority Orchestrator: reference clusters and tags for the demo backlog
# ---------------------------------------------------------------------------------------------

ORCHESTRATOR_CLUSTERS = [
    dict(member_ids=["eos-gid-paid_search-1", "eos-gid-all-2"],
         canonical_title="Clarify pricing for paid search evaluators",
         rationale="Both describe paid search evaluators leaving an unclear pricing page and propose the same plan comparison."),
    dict(member_ids=["eos-gid-paid_search-2", "eos-gid-all-3"],
         canonical_title="Fix the mobile demo form",
         rationale="Both describe the same broken mobile demo form and propose the same technical fix."),
]
ORCHESTRATOR_TAGS = {
    "Unclear pricing is losing paid search evaluators": (
        ["design", "content", "engineering"],
        "Whether pricing clarity, not traffic quality, drives the paid search decline.",
        "Demo requests per paid search pricing-page session."),
    "A broken mobile demo form loses ready-to-buy visitors": (
        ["engineering", "data"],
        "How much demand the mobile defect was costing, once completion tracking is verified.",
        "Mobile demo-form completion rate."),
    "Webinar leads are learners, not buyers": (
        ["sales", "operations", "content"],
        "Whether intent screening raises webinar lead quality without shrinking real demand.",
        "Webinar MQL to SQL conversion."),
    "Rising bids are buying lower-intent paid search traffic": (
        ["data"],
        "Whether keyword mix shifted toward low-intent queries as bids rose.",
        "Paid search lead to MQL conversion by keyword group."),
    "Slow partner follow-up loses partner demand": (
        ["operations", "sales"],
        "Whether faster partner follow-up changes acceptance or win rates.",
        "Partner MQL to SQL conversion by response time."),
}


def build_orchestrator(live: bool) -> None:
    import tempfile

    from gios.core.store import Store
    from gios.modules.demo_flow import seed_through
    from gios.modules.growth_priority_orchestrator import pipeline
    from gios.modules.growth_priority_orchestrator.models import ItemTags, OrchestratorProposal

    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp) / "seed.db")
        seed_through("backlog", store)
        inputs = pipeline.gather(store)
    if live:
        result = pipeline.propose(inputs.candidates, inputs.hypotheses)
        if result.proposal is None:
            raise SystemExit(result.error)
        proposal = result.proposal
    else:
        ids = {o.id for o in inputs.candidates}
        for c in ORCHESTRATOR_CLUSTERS:
            if not set(c["member_ids"]) <= ids:
                raise SystemExit(f"demo backlog changed; cluster ids missing: {c['member_ids']}")
        proposal = OrchestratorProposal(
            clusters=ORCHESTRATOR_CLUSTERS,
            items=[ItemTags(item_id=o.id, dependencies=ORCHESTRATOR_TAGS[o.title][0],
                            expected_learning=ORCHESTRATOR_TAGS[o.title][1], primary_metric=ORCHESTRATOR_TAGS[o.title][2])
                   for o in sorted(inputs.candidates, key=lambda o: o.id)])
    _write(pipeline.PROMPT, proposal)


# ---------------------------------------------------------------------------------------------
# Experiment Learning Capture: reference drafts per past experiment
# ---------------------------------------------------------------------------------------------


def _nh(title, problem, audience, cause, intervention, behavior, outcome):
    return dict(title=title, observed_problem=problem, affected_audience=audience, causal_explanation=cause,
                intervention=intervention, expected_behavior_change=behavior, expected_business_outcome=outcome)


LEARNING_REFERENCE = {
    "EXP-07": dict(
        theme="pricing_uncertainty",
        original_problem="Paid search visitors leave the pricing page; we tried a discount offer as they exit.",
        what_happened=("The popup appeared to lift demo requests sharply, but the test was small, short, and ended "
                       "weeks before any of those leads could reach a decision."),
        learned_about_customer="A price incentive catches attention, but we do not know whether it attracts buyers or bargain hunters.",
        learned_about_journey="The exit moment on pricing is reachable; what visitors need there is still unclear.",
        learned_about_business="Discounts can buy top-funnel lifts that quality and margin later take back.",
        should_not_conclude=("That the popup works. A large lift on a tiny sample over two weeks is the classic "
                             "outlier; downstream outcomes have not matured and the effect differs by source."),
        reusable_principle="Treat a surprisingly large lift on a small, short test as a reason to retest, not to ship.",
        next_hypothesis=_nh("Retest the exit offer at full size",
                            "A small test showed a large but immature lift from an exit discount.",
                            "Paid search pricing-page visitors",
                            "A price incentive at the exit moment may convert hesitant evaluators.",
                            "Rerun the popup on full paid search traffic for a full sales cycle with a no-discount "
                            "clarity message as a third arm.",
                            "Demo requests rise without a fall in sales acceptance.",
                            "Lower paid search acquisition cost without discount-driven churn."),
    ),
    "EXP-01": dict(
        theme="unclear_value",
        original_problem="Homepage visitors were not requesting demos; feature-led copy did not explain the outcome.",
        what_happened=("The outcome headline lifted demo requests, and the extra requests carried through to more "
                       "qualified pipeline rather than fading after the form."),
        learned_about_customer="Evaluators respond to the job the product does for them, not to a feature list.",
        learned_about_journey="Clarity on the first screen shapes intent all the way to qualification.",
        learned_about_business="Homepage messaging is a cheap lever on qualified pipeline, not only on lead volume.",
        should_not_conclude=("That any shorter or punchier headline will work; this tested one specific outcome "
                             "message. Nor that win rates rose: wins were too few to test."),
        reusable_principle="Lead with the customer outcome, then prove it with features.",
        next_hypothesis=_nh("Carry the outcome message into paid search landing pages",
                            "Paid search visitors land on feature-led pages and convert below benchmark.",
                            "Paid search evaluators", "The outcome message that worked on the homepage is missing "
                            "where paid traffic lands.", "Use the outcome headline on paid search landing pages.",
                            "More paid search visitors request a demo.", "Lower paid search acquisition cost."),
    ),
    "EXP-02": dict(
        theme="urgency",
        original_problem="Webinar registration felt heavy, so we tried a shorter form to grow attendance.",
        what_happened=("The short form brought in many more registrations and MQLs, but sales accepted a much "
                       "smaller share of them, so qualified pipeline did not grow."),
        learned_about_customer=("Removing qualifying questions lets in curious learners who have no budget or "
                                "active project."),
        learned_about_journey="Friction at registration was doing useful qualification work before the sales handoff.",
        learned_about_business="Optimizing registrations alone shifts cost to sales without adding pipeline.",
        should_not_conclude=("That shorter forms are bad in general, or that webinars do not work. The loss came "
                             "from removing intent questions, not from form length itself."),
        reusable_principle="Measure a form change at the stage where quality is decided, not where volume is counted.",
        next_hypothesis=_nh("Short form plus one intent question",
                            "Short registration forms raise volume but lower sales acceptance.",
                            "Webinar registrants", "Without any intent signal, learners and buyers look the same.",
                            "Keep the short form but add a single buying-timeline question that routes non-buyers to "
                            "nurture.", "Registrations stay high while sales acceptance recovers.",
                            "More qualified webinar pipeline per sales hour."),
    ),
    "EXP-03": dict(
        theme="pricing_uncertainty",
        original_problem="Paid search visitors exit the pricing page; we guessed unanswered plan questions were the cause.",
        what_happened=("The FAQ accordion moved click-through slightly, but the test was too small and too short "
                       "to tell a real effect from noise."),
        learned_about_customer="We still do not know which plan details confuse evaluators.",
        learned_about_journey="Pricing questions are not resolved by content hidden below the pricing table.",
        learned_about_business="Small pricing tests on paid search traffic need several weeks to reach a verdict.",
        should_not_conclude=("That pricing clarity does not matter. This test could only detect a large effect, "
                             "and an FAQ is a weak version of the fix."),
        reusable_principle="Size the test before running it; an underpowered test teaches nothing.",
        next_hypothesis=_nh("Plan comparison table for paid search",
                            "Paid search evaluators exit the pricing page at a rising rate.",
                            "Paid search evaluators", "Plan inclusions and limits are hard to compare.",
                            "Replace the plan cards with a comparison table of inclusions and limits.",
                            "Fewer exits and more demo clicks from the pricing page.",
                            "Lower paid search acquisition cost."),
    ),
    "EXP-04": dict(
        theme="trust_proof",
        original_problem="Demo-page visitors hesitate to start the form; we tested social proof.",
        what_happened=("Customer logos lifted form starts on desktop but not on mobile, and the extra starts did "
                       "not turn into more qualified pipeline."),
        learned_about_customer="Desktop evaluators notice and value logos; mobile visitors do not see them in the same way.",
        learned_about_journey="On mobile the demo page has a bigger problem than trust: the form itself.",
        learned_about_business="Top-funnel lifts that stop at the form start do not pay back on their own.",
        should_not_conclude=("That social proof does not work on mobile; the mobile form is broken, which masks "
                             "any trust effect there."),
        reusable_principle="Fix broken steps before testing persuasion on the same step.",
        next_hypothesis=_nh("Retest logos after the mobile form fix",
                            "Logos lifted desktop starts only.", "Mobile demo-page visitors",
                            "The broken mobile form hides any effect of social proof.",
                            "Rerun the logo test once mobile form completion is repaired.",
                            "Mobile starts and completions both rise.", "More demo requests from mobile traffic."),
    ),
    "EXP-05": dict(
        theme="unclear_value",
        original_problem="Paid social traffic arrives cold, so we tried a long-form landing page to educate it.",
        what_happened="The long-form page lowered lead conversion clearly, with no offsetting gain in quality.",
        learned_about_customer="Cold social visitors will not read a long page before deciding.",
        learned_about_journey="For cold traffic the first step must be small; education belongs after the first conversion.",
        learned_about_business="Paid social spend is wasted when the landing page asks for too much attention.",
        should_not_conclude=("That paid social cannot work for EchoAI, or that all long pages fail. This was one "
                             "page for cold traffic."),
        reusable_principle="Match the ask to the visitor's temperature.",
        next_hypothesis=_nh("Lighter first step for paid social",
                            "Cold paid social visitors rarely convert.", "Paid social visitors",
                            "A demo request is too large a first ask for cold traffic.",
                            "Offer a short product tour as the first step.", "More paid social visitors engage.",
                            "More nurtured leads from paid social at the same spend."),
    ),
    "EXP-06": dict(
        theme="complexity",
        original_problem="Nurture emails asked for a demo, a big commitment for leads still learning.",
        what_happened=("The two-minute tour lifted clicks to leads, but too little time has passed for these "
                       "leads to reach a decision."),
        learned_about_customer="Leads in nurture prefer a small, low-commitment next step.",
        learned_about_journey="A lighter step widens the funnel; whether it fills pipeline is not yet known.",
        learned_about_business="Email changes are cheap, so waiting for downstream results costs little.",
        should_not_conclude=("That the tour improves revenue or win rates. Downstream outcomes have not matured, "
                             "and early volume gains can hide later quality losses."),
        reusable_principle="Judge funnel changes over the full sales cycle, not the first weeks.",
        next_hypothesis=_nh("Tour CTA quality check after a full sales cycle",
                            "The tour CTA lifts early engagement.", "Nurture email leads",
                            "A smaller ask brings forward leads at every intent level.",
                            "Keep the test running and compare qualification after a full cycle.",
                            "Sales acceptance of tour leads matches demo-CTA leads.",
                            "More qualified pipeline from nurture."),
    ),
}


def build_learning_capture(live: bool) -> None:
    import tempfile

    from gios.core.schemas import Experiment
    from gios.core.store import Store
    from gios.modules.downstream_impact_analyzer import analyze_all
    from gios.modules.experiment_learning_capture import pipeline
    from gios.modules.experiment_learning_capture.models import LearningDraft

    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp) / "seed.db")
        analyze_all(store)
        experiments = store.list(Experiment)
    for e in experiments:
        d = pipeline.draft(e) if live else LearningDraft(**LEARNING_REFERENCE[e.id])
        _write(f"{pipeline.PROMPT}__{e.id}", d)


# ---------------------------------------------------------------------------------------------
# Growth Council Insight Brief: reference brief for the full period
# ---------------------------------------------------------------------------------------------

BRIEF_REFERENCE = dict(
    what_changed=("Paid search is getting sharply more expensive per customer. At the same time, paid search "
                  "visitors are leaving the pricing page more often every month, and evaluators increasingly tell "
                  "us they cannot work out what the plans include."),
    why_it_matters=("Paid search is our largest paid channel. If unclear pricing is turning high-intent searchers "
                    "away, every extra dollar of bid spend buys less pipeline, and the fix is on our own page rather "
                    "than in the auction."),
    signals=dict(
        customer=dict(text="Pricing-clarity comments are the top-ranked customer theme and are rising; one names "
                           "arriving from a search ad.", ids=["css-pricing_uncertainty"]),
        behavioral=dict(text="Paid search exits on the pricing page are far above other sources and climbing, with "
                             "fewer demo clicks.", ids=["bfa-pricing-source-paid_search"]),
        funnel=dict(text="Fewer paid search visits become leads than for other sources.",
                    ids=["qdl-leak-paid_search-visit-to-lead"]),
        revenue=dict(text="Paid search acquisition cost rose steeply as spend grew while wins fell.",
                     ids=["qdl-flag-rising_cac-paid_search"]),
        sales_operational=dict(text="No sales or operational signal points at paid search; follow-up speed is "
                                    "normal for these leads.", ids=[]),
    ),
    working_hypothesis=("Unclear plan inclusions and limits on the pricing page are losing paid search evaluators "
                        "before they request a demo."),
    hypothesis_id="gid-paid_search-1",
    recommended_action=("Run the pricing-page experiment for paid search traffic: a clear plan comparison with "
                        "inclusions and usage limits, plus a short intercept survey on what is unclear."),
    opportunity_id="eos-gid-all-2",
    cross_functional_dependency=("Web and design build the comparison; product marketing owns plan wording; paid "
                                 "search holds bids steady during the test so traffic quality does not shift."),
    owner="web_conversion",
    owner_rationale="The loss happens on our pricing page, between the visit and the lead.",
    measurement=("Primary: demo requests per paid search pricing-page session. Guardrail: lead to SQL rate for paid "
                 "search. Downstream: paid search acquisition cost over the following quarter."),
    organization_learning=("Our earlier FAQ test on this page was too small to read, and the webinar form test "
                           "showed that volume gains can hide quality losses. Size this test properly and judge it on "
                           "qualified pipeline, not clicks."),
    learning_ids=["elc-EXP-03", "elc-EXP-02"],
    decision_needed=("Approve the pricing-page experiment as a Now priority, and agree to hold paid search bids "
                     "flat until it reads out."),
)


def build_growth_council_brief(live: bool) -> None:
    import tempfile

    from gios.core.store import Store
    from gios.modules.demo_flow import seed_through
    from gios.modules.growth_council_insight_brief import analysis, pipeline
    from gios.modules.growth_council_insight_brief.models import checked_brief_model

    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp) / "seed.db")
        seed_through("learnings", store)
        inputs = analysis.gather(store, *pipeline.FULL_PERIOD)
        draft = (pipeline.write(inputs).draft if live
                 else checked_brief_model(inputs.allowed_ids())(**BRIEF_REFERENCE))
    _write(f"{pipeline.PROMPT}__{pipeline.demo_key(*pipeline.FULL_PERIOD)}", draft)


# ---------------------------------------------------------------------------------------------


WRITTEN: set[str] = set()


def _write(name: str, obj: BaseModel) -> None:
    WRITTEN.add(f"{name}.json")
    path = config.DEMO_DIR / f"{name}.json"
    payload: Any = json.loads(obj.model_dump_json())
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    type(obj).model_validate_json(path.read_text())
    print(f"wrote {path}")


BUILDERS: dict[str, Callable[[bool], None]] = {
    "customer_signal": build_customer_signal,
    "behavioral_friction": build_behavioral_friction,
    "demand_leakage": build_demand_leakage,
    "growth_diagnostic": build_growth_diagnostic,
    "hypothesis_validator": build_hypothesis_validator,
    "orchestrator": build_orchestrator,
    "learning_capture": build_learning_capture,
    "growth_council_brief": build_growth_council_brief,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build demo/ cached LLM outputs.")
    parser.add_argument("--live", action="store_true", help="call the real model (needs ANTHROPIC_API_KEY)")
    parser.add_argument("--only", choices=sorted(BUILDERS), help="build one module only")
    args = parser.parse_args()
    if args.live and config.is_demo_mode():
        parser.error("--live needs ANTHROPIC_API_KEY")
    if not args.live:
        import os

        os.environ.pop("ANTHROPIC_API_KEY", None)  # offline builds must never call the API
    for name, builder in BUILDERS.items():
        if args.only in (None, name):
            builder(args.live)
    if args.only is None:  # a full build owns demo/: drop outputs it no longer produces
        for stale in sorted(p for p in config.DEMO_DIR.glob("*.json") if p.name not in WRITTEN):
            stale.unlink()
            print(f"removed stale {stale.name}")


if __name__ == "__main__":
    main()
