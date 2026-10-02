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
# Behavioral Friction Analyzer: reference classifications for default-threshold findings
# ---------------------------------------------------------------------------------------------

FRICTION_REFERENCE = {
    "demo|device=mobile": dict(
        friction_type="technical",
        what_happened=("Mobile visitors start the demo form at least as often as desktop visitors but "
                       "complete it far less often, rage-click much more, and wait much longer for the page to load."),
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
        "channel, but sales accepts very few of those MQLs. The loss happens at the MQL to SQL handoff, and "
        "sales rejection reasons point to a qualification mismatch: attendees without budget, authority, or "
        "an active project. Follow-up speed for webinar leads is in line with other sources, so this is not an "
        "SLA problem. Separately, partner leads wait longer for first contact and a small share of MQLs is "
        "never routed."),
    probable_causes=[
        dict(owner="qualification", explanation=(
            "The MQL definition rewards webinar attendance and engagement, which inflates scores for learners and "
            "researchers who lack budget or authority; sales then rejects them.")),
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
                causal_explanation=("The MQL rules reward webinar attendance, so contacts without budget, authority, "
                                    "or an active project reach sales."),
                intervention="Add buying-intent questions to registration and require an intent signal before MQL.",
                expected_behavior_change="Fewer but better webinar MQLs; sales accepts a much larger share.",
                expected_business_outcome="More SQLs per sales hour and steady webinar-sourced pipeline.",
                supporting=[
                    _claim("qdl-leak-webinar-mql-to-sql", "Webinar MQL to SQL conversion is far below other sources "
                           "on large volume."),
                    _claim("qdl-flag-qualification_mismatch-webinar", "Most webinar rejections cite budget, "
                           "authority, or student status."),
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

from gios.modules.demo_flow import DIAGNOSTIC_SCOPES, diagnostic_inputs  # noqa: E402


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
    for scope in DIAGNOSTIC_SCOPES:
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


def build_hypothesis_validator(live: bool) -> None:
    from gios.modules.growth_intelligence_diagnostic import pipeline as gid
    from gios.modules.hypothesis_evidence_validator import analysis, pipeline
    from gios.modules.hypothesis_evidence_validator.examples import ASSUMPTION_ONLY
    from gios.modules.hypothesis_evidence_validator.models import DimensionScore, ValidatorProposal

    signals = _phase1_signals()
    hypotheses = {ASSUMPTION_ONLY.fingerprint(): ASSUMPTION_ONLY}
    for scope in DIAGNOSTIC_SCOPES:
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


def _write(name: str, obj: BaseModel) -> None:
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


if __name__ == "__main__":
    main()
