import json

import numpy as np
import pandas as pd
import pytest

from gios import config
from gios.core import data
from gios.core.schemas import Signal
from gios.core.store import Store
from gios.modules.behavioral_friction_analyzer import Thresholds, analysis, pipeline, run, to_signals
from gios.modules.behavioral_friction_analyzer.models import FrictionBatch, FrictionClassification
from gios.modules.behavioral_friction_analyzer.report import to_markdown
from gios.modules.customer_signal_synthesizer import analysis as css_analysis
from gios.modules.customer_signal_synthesizer import run as css_run
from tests.conftest import FakeClient


def web_frame(cells, months=("2026-03",)):
    """cells: list of dicts with page, source, device and counts; repeated per month."""
    rows = []
    for m in months:
        for c in cells:
            base = dict(date=f"{m}-01", page="home", source="organic", device="desktop", sessions=1000, exits=400,
                        scroll_75=300, cta_clicks=120, form_starts=0, form_completes=0, rage_clicks=4,
                        avg_load_ms=1300)
            base.update(c)
            rows.append(base)
    df = pd.DataFrame(rows)
    df["month"] = df.date.str[:7]
    return df


def funnel_frame():
    return pd.DataFrame([
        dict(month="2026-03", source=s, segment="smb", visits=10000, leads=300, mqls=150, sqls=60, opps=30,
             wins=10, revenue=100000, spend=1) for s in ["organic", "paid_search", "paid_social"]])


# --- stats on slices ---


def test_standardized_comparison_removes_mix_effect():
    # paid_social is mostly mobile; mobile completes worse for EVERY source.
    cells = pd.DataFrame([
        dict(source="paid_social", device="mobile", form_completes=200, form_starts=1000),
        dict(source="paid_social", device="desktop", form_completes=60, form_starts=100),
        dict(source="organic", device="mobile", form_completes=20, form_starts=100),
        dict(source="organic", device="desktop", form_completes=600, form_starts=1000),
    ])
    x, n, obs, exp = analysis.standardized_comparison(cells, "source", "paid_social", "device",
                                                      "form_completes", "form_starts")
    assert (x, n) == (260, 1100)
    assert obs == pytest.approx(260 / 1100)
    assert exp == pytest.approx((1000 * 0.2 + 100 * 0.6) / 1100)  # same as observed: no source effect
    raw_rest = 620 / 1100
    assert obs < raw_rest  # a naive comparison would have flagged paid_social


def test_divergence_flags_only_bad_direction_and_respect_thresholds():
    web = web_frame([
        dict(page="demo", device="desktop", sessions=5000, form_starts=1500, form_completes=900),
        dict(page="demo", device="mobile", sessions=5000, form_starts=1800, form_completes=360),
        dict(page="home", device="desktop", sessions=5000, exits=2000),
        dict(page="home", device="mobile", sessions=5000, exits=1500),  # better: never flagged
    ])
    flags = analysis.divergence_flags(web, Thresholds())
    assert ("demo", "device", "mobile") in flags
    assert [f.metric for f in flags[("demo", "device", "mobile")]] == ["form_completion"]
    assert ("home", "device", "mobile") not in flags
    # desktop home exits are higher than mobile: that IS the bad direction for desktop
    assert flags[("home", "device", "desktop")][0].metric == "exit_rate"
    assert not analysis.divergence_flags(web, Thresholds(min_volume=10_000))
    assert not analysis.divergence_flags(web, Thresholds(min_deviation=0.9))


def test_load_flags_vs_same_device_site_baseline():
    web = web_frame([
        dict(page="demo", device="mobile", avg_load_ms=4800),
        dict(page="home", device="mobile", avg_load_ms=2000),
        dict(page="pricing", device="mobile", avg_load_ms=2200),
        dict(page="demo", device="desktop", avg_load_ms=1400),
        dict(page="home", device="desktop", avg_load_ms=1300),
    ])
    flags = analysis.load_flags(web, Thresholds())
    assert list(flags) == [("demo", "device", "mobile")]
    f = flags[("demo", "device", "mobile")][0]
    assert f.expected == pytest.approx(2100) and f.deviation == pytest.approx(4800 / 2100 - 1)


def test_trend_flags_worsening_only():
    first = web_frame([dict(page="pricing", source="paid_search", exits=450),
                       dict(page="pricing", source="organic", exits=400)], months=["2026-03"])
    last = web_frame([dict(page="pricing", source="paid_search", exits=700),
                      dict(page="pricing", source="organic", exits=300)], months=["2026-08"])
    flags = analysis.trend_flags(pd.concat([first, last]), Thresholds())
    assert list(flags) == [("pricing", "source", "paid_search")]
    assert flags[("pricing", "source", "paid_search")][0].observed == pytest.approx(0.7)


def test_sizing_uses_largest_metric_and_propagates():
    funnel = funnel_frame()
    rates = analysis.downstream_rates(funnel)
    assert rates.loc["organic", "lead_to_win"] == pytest.approx(10 / 300)
    f = analysis.Finding("x", "demo", "device", "mobile", flags=[
        analysis.MetricFlag("device", "form_completion", "l", 0.2, 0.6, -0.67, -10, 600, 120),
        analysis.MetricFlag("load", "load_ms", "l", 4800, 2100, 1.3, None, 6000),
    ], context={"source_mix": {"organic": 1.0}, "months": 2})
    s = analysis.size_finding(f, rates, demo_completion_rate=0.1)
    assert s["metric"] == "form_completion"
    assert s["lost_leads_per_month"] == pytest.approx(0.4 * 600 / 2)
    assert s["lost_wins_per_month"] == pytest.approx(120 * 10 / 300)
    assert s["lost_revenue_per_month"] == pytest.approx(4 * 10000)
    unsized = analysis.Finding("y", "demo", "device", "mobile", flags=[f.flags[1]], context={"months": 1})
    assert analysis.size_finding(unsized, rates, 0.1) == {"sized": False}


# --- LLM step and evidence guard ---


def classification(fid, ids=()):
    return FrictionClassification(
        finding_id=fid, friction_type="technical", what_happened="w", competing_explanations=["a", "b"],
        validation_needed=["v"], supporting_customer_signal_ids=list(ids), recommended_action="technical_fix",
        recommended_detail="d")


def css_signal(sid):
    return Signal(id=sid, type="customer", evidence_status="observed", source="x", segment="all",
                  journey_stage="evaluation", summary="s", strength=3)


def test_evidence_guard_never_trusts_unknown_ids():
    c = classification("f", ids=["css-real", "css-invented"])
    check = pipeline.check_evidence(c, [css_signal("css-real")])
    assert check.linked_signal_ids == ["css-real"] and check.dropped_signal_ids == ["css-invented"]
    assert pipeline.check_evidence(c, []).status == pipeline.UNCONFIRMED


def test_classify_live_filters_to_requested_findings(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    f = analysis.Finding("demo|device=mobile", "demo", "device", "mobile", flags=[
        analysis.MetricFlag("device", "form_completion", "Form", 0.2, 0.6, -0.67, -10.0, 600, 120)])
    reply = FrictionBatch(classifications=[classification("demo|device=mobile"),
                                           classification("other")]).model_dump_json()
    client = FakeClient([reply])
    out = pipeline.classify([f], [], client=client)
    assert list(out) == ["demo|device=mobile"]
    assert "Do not cite any customer signal ids" in client.calls[0]["messages"][0]["content"]


# --- acceptance on the synthetic data ---


@pytest.fixture(scope="module")
def demo_result():
    return run(customer_signals=css_analysis.to_signals(css_run().themes))


def test_acceptance_mobile_demo_form_is_technical_friction(demo_result):
    ids = [f.id for f in demo_result.findings]
    assert ids[0] == "demo|device=mobile"
    f = demo_result.findings[0]
    assert {fl.metric for fl in f.flags} >= {"form_completion", "rage_rate", "load_ms"}
    assert demo_result.classifications[f.id].friction_type == "technical"
    assert demo_result.evidence[f.id].linked_signal_ids == ["css-technical_issue"]
    assert f.sizing["sized"] and f.sizing["lost_leads_per_month"] > 500


def test_acceptance_paid_search_pricing_exits(demo_result):
    f = next(f for f in demo_result.findings if f.id == "pricing|source=paid_search")
    metrics = {(fl.check, fl.metric) for fl in f.flags}
    assert ("source", "exit_rate") in metrics and ("trend", "exit_rate") in metrics
    assert demo_result.evidence[f.id].status == "supported by customer evidence"


def test_no_spurious_findings_on_other_pages(demo_result):
    pages = {(f.page, f.value) for f in demo_result.findings if f.max_abs_z > 10}
    assert pages == {("demo", "mobile"), ("pricing", "paid_search")}


def test_without_customer_signals_causes_are_unconfirmed():
    result = run()
    assert all(e.status == pipeline.UNCONFIRMED for e in result.evidence.values())
    md = to_markdown(result)
    assert "No customer evidence is linked yet" in md
    assert "supported by customer evidence" not in md


def test_demo_cache_covers_default_findings():
    cached = FrictionBatch.model_validate_json((config.DEMO_DIR / "classify_behavioral_friction.json").read_text())
    assert {c.finding_id for c in cached.classifications} == {f.id for f in run().findings}


def test_unclassified_findings_render_placeholder():
    result = run(thresholds=Thresholds(z=1.0, min_deviation=0.02, load_deviation=0.05))
    assert len(result.findings) > 3
    assert "Not classified" in to_markdown(result)


def test_markdown_and_signals(demo_result, store):
    md = to_markdown(demo_result)
    for heading in ["### Behavioral Signal", "### Friction Type", "### Affected Segment",
                    "### Likely Business Impact", "### Competing Explanations", "### Validation Needed",
                    "### Recommended Experiment / Research"]:
        assert md.count(heading) == len(demo_result.findings)
    signals = to_signals(demo_result)
    assert {s.type for s in signals} == {"behavioral"}
    by_id = {s.id: s for s in signals}
    assert by_id["bfa-demo-device-mobile"].strength == 5
    assert by_id["bfa-compare-device-desktop"].strength <= 2
    store.replace_module_output(signals, "behavioral_friction_analyzer")


def test_page_renders_and_saves():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(config.ROOT / "pages" / "2_Behavioral_Friction_Analyzer.py")).run(timeout=60)
    assert not at.exception
    assert any("Finding 1: Demo page, mobile visitors" in m.value for m in at.markdown)
    at.button(key="bfa_save").click().run(timeout=60)
    assert len(Store().list(Signal, module="behavioral_friction_analyzer")) == 3
