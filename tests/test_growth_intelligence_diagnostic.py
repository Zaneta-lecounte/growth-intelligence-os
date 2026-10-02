import json

import pytest
from pydantic import ValidationError

from gios import config
from gios.core import data
from gios.core.llm import LLMError
from gios.core.schemas import Hypothesis, Recommendation, Signal
from gios.core.store import Store
from gios.modules.growth_intelligence_diagnostic import (
    BusinessSignal,
    Filters,
    analysis,
    missing_business_fields,
    pipeline,
    run,
)
from gios.modules.growth_intelligence_diagnostic.models import (
    DiagnosisOutput,
    EvidenceClaim,
    RootCause,
    checked_output_model,
)
from gios.modules.growth_intelligence_diagnostic.report import to_markdown
from gios.modules.signal_layer import run_signal_layer
from tests.conftest import FakeClient


def sig(sid, type_="customer", status="observed", strength=3, channel=None, segment="all", period="2026-03..2026-08"):
    return Signal(id=sid, type=type_, evidence_status=status, source="x", segment=segment, journey_stage="evaluation",
                  summary=sid, strength=strength, channel=channel, period=period)


def claim(sid, status="observed"):
    return EvidenceClaim(signal_id=sid, claim="c", status=status)


def root(supporting, contradicting=()):
    return RootCause(title="t", observed_problem="p", affected_audience="a", causal_explanation="c",
                     intervention="i", expected_behavior_change="b", expected_business_outcome="o",
                     supporting=list(supporting), contradicting=list(contradicting))


def business(**kw):
    base = dict(goal="Grow pipeline", target_metric="CAC", what="Paid search CAC rose",
                baseline="Earlier half", segment="all", period="2026-03 to 2026-08")
    return BusinessSignal(**(base | kw))


@pytest.fixture(scope="module")
def phase1_signals(tmp_path_factory):
    store = Store(tmp_path_factory.mktemp("p1") / "s.db")
    run_signal_layer(store)
    return store.list(Signal)


# --- Step 1 gate ---


def test_business_signal_requires_every_field():
    assert missing_business_fields({}) == list(
        ["Business goal", "Target metric", "What is underperforming", "Baseline", "Audience / segment", "Period"])
    full = business().model_dump()
    assert missing_business_fields(full) == []
    assert missing_business_fields(full | {"baseline": "  "}) == ["Baseline"]
    with pytest.raises(ValidationError):
        business(baseline="")


def test_suggest_business_signal_is_computed_from_data():
    funnel = data.load("funnel_by_source")
    draft = analysis.suggest_business_signal(funnel, "CAC", "paid_search", None, "2026-03", "2026-08")
    h = analysis.metric_halves(funnel, "CAC", "paid_search", None, "2026-03", "2026-08")
    f = funnel[funnel.source == "paid_search"]
    early = f[f.month <= "2026-05"]
    assert h["early"] == pytest.approx(early.spend.sum() / early.wins.sum())
    assert h["worse"] and h["change"] > 0.25
    assert "worsened" in draft["what"] and draft["period"] == "2026-03 to 2026-08"
    assert missing_business_fields(draft | {"goal": "Grow"}) == []


# --- Steps 2–4 ---


def test_filter_signals_by_channel_segment_period():
    signals = [sig("a", channel="paid_search"), sig("b", channel="webinar"), sig("c"),
               sig("d", segment="smb"), sig("e", segment="mid_market"), sig("f", period="2025-01..2025-06")]
    ids = lambda xs: sorted(s.id for s in xs)  # noqa: E731
    assert ids(analysis.filter_signals(signals, channel="paid_search")) == ["a", "c", "d", "e", "f"]
    assert ids(analysis.filter_signals(signals, segment="smb")) == ["a", "b", "c", "d", "f"]
    assert ids(analysis.filter_signals(signals, start="2026-03", end="2026-08")) == ["a", "b", "c", "d", "e"]


def test_classify_covers_all_six_types():
    out = analysis.classify([sig("a", "revenue"), sig("b", "operational", strength=5), sig("c", "operational")])
    assert list(out) == analysis.SIGNAL_TYPES
    assert [s.id for s in out["operational"]] == ["b", "c"]


def test_leakage_point_is_deterministic_auditor_output():
    funnel, sales = data.load("funnel_by_source"), data.load("sales_feedback")
    overall = analysis.leakage_point(funnel, sales, None, None, "2026-03", "2026-08")
    assert (overall.source, overall.stage) == ("webinar", "MQL→SQL")
    ps = analysis.leakage_point(funnel, sales, "paid_search", None, "2026-03", "2026-08")
    assert ps.source == "paid_search" and ps.stage == "Visit→Lead"
    assert analysis.leakage_point(funnel, sales, None, None, "2027-01", "2027-02") is None


# --- Step 5: id validation, ranking, confidence ---


def test_checked_model_rejects_unknown_signal_ids():
    model = checked_output_model(frozenset({"a", "b"}))
    payload = dict(growth_problem="g", hypotheses=[root([claim("a")]).model_dump(), root([claim("b")]).model_dump()],
                   next_best_action="experiment", action_detail="d", primary_metric="p", guardrail_metric="g",
                   downstream_metric="d", learning_objective="l")
    assert model(**payload)
    payload["hypotheses"][1]["contradicting"] = [claim("ghost").model_dump()]
    with pytest.raises(ValidationError, match="ghost"):
        model(**payload)


def test_output_requires_two_to_five_hypotheses():
    base = dict(growth_problem="g", next_best_action="experiment", action_detail="d", primary_metric="p",
                guardrail_metric="g", downstream_metric="d", learning_objective="l")
    with pytest.raises(ValidationError):
        DiagnosisOutput(**base, hypotheses=[root([claim("a")])])
    with pytest.raises(ValidationError):
        DiagnosisOutput(**base, hypotheses=[root([claim("a")])] * 6)
    with pytest.raises(ValidationError):  # narrative fields carry no numbers
        DiagnosisOutput(**(base | {"growth_problem": "CAC up 76%"}), hypotheses=[root([claim("a")])] * 2)


def test_ranking_and_confidence_rules():
    by_id = {s.id: s for s in [sig("c1", "customer", strength=5), sig("b1", "behavioral", strength=4),
                               sig("a1", "acquisition", strength=4), sig("f1", "funnel", status="inferred"),
                               sig("o1", "operational", strength=3)]}
    strong = root([claim("c1"), claim("b1"), claim("a1")])
    two_types = root([claim("c1"), claim("f1")])
    contradicted = root([claim("c1"), claim("b1"), claim("a1")], [claim("o1")])
    ranked = analysis.rank_hypotheses([two_types, contradicted, strong], by_id)
    assert [r.score for r in ranked] == [13, 10, 8]
    assert [r.confidence for r in ranked] == ["high", "medium", "medium"]
    assert analysis.confidence(root([claim("c1")]), by_id) == "low"
    # an inferred signal can only support an inferred claim
    assert ranked[2].root_cause.supporting[1].status == "inferred"


def test_live_output_citing_unknown_ids_is_rejected_after_one_retry(monkeypatch, phase1_signals):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    cached = json.loads((config.DEMO_DIR / "run_growth_diagnostic__paid_search.json").read_text())
    cached["hypotheses"][0]["supporting"][0]["signal_id"] = "css-invented"
    client = FakeClient([json.dumps(cached), json.dumps(cached)])
    with pytest.raises(LLMError, match="css-invented"):
        run(business(), Filters(channel="paid_search"), phase1_signals, client=client)
    assert len(client.calls) == 2
    assert "css-invented" in client.calls[1]["messages"][2]["content"]


def test_demo_output_rejected_when_scope_excludes_cited_signals(phase1_signals):
    # The cached paid-search diagnosis cites css-pricing_uncertainty (segment smb); mid-market excludes it.
    with pytest.raises(LLMError, match="do not exist"):
        run(business(), Filters(channel="paid_search", segment="mid_market"), phase1_signals)


# --- acceptance ---


@pytest.fixture(scope="module")
def paid_search_result(phase1_signals):
    return run(business(), Filters(channel="paid_search"), phase1_signals)


def test_acceptance_paid_search_top_hypothesis_links_three_signals(paid_search_result):
    top = paid_search_result.top
    cited = {c.signal_id for c in top.root_cause.supporting}
    assert {"css-pricing_uncertainty", "bfa-pricing-source-paid_search", "qdl-flag-rising_cac-paid_search"} <= cited
    assert {"customer", "behavioral", "acquisition"} <= set(top.supporting_types)
    assert top.confidence == "high"
    assert paid_search_result.leakage_point.source == "paid_search"


def test_all_channels_top_hypothesis_is_webinar(phase1_signals):
    r = run(business(), Filters(), phase1_signals)
    assert "qdl-leak-webinar-mql-to-sql" in {c.signal_id for c in r.top.root_cause.supporting}
    assert r.leakage_point.stage == "MQL→SQL" and r.output.next_best_action == "process_change"
    # a hypothesis whose only evidence is also contradicted ranks last with low confidence
    assert r.ranked[-1].confidence == "low"


def test_markdown_matches_spec_format(paid_search_result):
    md = to_markdown(paid_search_result)
    headings = [l for l in md.splitlines() if l.startswith("### ")]
    assert headings == ["### Growth Problem", "### Evidence", "### Likely Root Causes",
                        "### Highest-Priority Opportunity", "### Recommended Action", "### Measurement Plan",
                        "### Confidence", "### Missing Evidence"]
    for sub in ["**Customer:**", "**Behavioral:**", "**Funnel:**", "**Business / Revenue:**"]:
        assert sub in md
    for prefix in ["- Primary:", "- Guardrail:", "- Downstream:", "- Learning objective:"]:
        assert prefix in md
    assert "`[Observed]`" in md and "`[Inferred]`" in md and "`[Unknown]`" in md
    assert ":green-badge[Observed]" in to_markdown(paid_search_result, ui=True)


def test_save_hypotheses_and_keep_scores(paid_search_result, store):
    pipeline.save(paid_search_result, store)
    hyps = {h.id: h for h in store.list(Hypothesis, module="growth_intelligence_diagnostic")}
    assert set(hyps) == {"gid-paid_search-1", "gid-paid_search-2", "gid-paid_search-3"}
    top = hyps["gid-paid_search-1"]
    assert top.confidence == "high" and "css-pricing_uncertainty" in top.signal_ids and not top.missing_parts
    assert store.get(Recommendation, "gid-paid_search-action").action_type == "experiment"
    top.evidence_diversity = 2
    store.save(top)
    pipeline.save(paid_search_result, store)  # unchanged content keeps validator scores
    assert store.get(Hypothesis, "gid-paid_search-1").evidence_diversity == 2


def test_page_gates_and_runs(monkeypatch):
    from streamlit.testing.v1 import AppTest

    run_signal_layer(Store())
    at = AppTest.from_file(str(config.ROOT / "views" / "4_Growth_Intelligence_Diagnostic.py")).run(timeout=60)
    assert not at.exception
    assert at.button(key="gid_run").disabled
    assert any("Diagnosis blocked" in e.value for e in at.error)
    at.selectbox(key="gid_channel").select("paid_search").run(timeout=60)
    at.selectbox(key="gid_fill_metric").select("CAC").run(timeout=60)
    at.button(key="gid_fill").click().run(timeout=60)
    assert not at.button(key="gid_run").disabled
    at.button(key="gid_run").click().run(timeout=60)
    assert not at.exception
    assert any("Unclear pricing is losing paid search evaluators" in m.value for m in at.markdown)
    at.button(key="gid_save").click().run(timeout=60)
    assert len(Store().list(Hypothesis)) == 3


def test_page_offers_seeding_when_store_empty():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(config.ROOT / "views" / "4_Growth_Intelligence_Diagnostic.py")).run(timeout=60)
    assert at.button(key="gid_seed")
    at.button(key="gid_seed").click().run(timeout=120)
    assert len(Store().list(Signal)) == 28
