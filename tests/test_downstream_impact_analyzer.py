import math
from datetime import date

import pandas as pd
import pytest

from gios import config
from gios.core import data, stats
from gios.core.schemas import Experiment
from gios.core.store import Store
from gios.modules.downstream_impact_analyzer import Settings, analysis, analyze_all, experiment_ids, pipeline, run
from gios.modules.downstream_impact_analyzer.examples import VOLUME_QUALITY_EXAMPLE
from gios.modules.downstream_impact_analyzer.report import to_markdown


def variants(c, t):
    cols = ["visitors", "conversions", "mqls", "sqls", "opps", "wins", "revenue", "spend"]
    return pd.DataFrame([c, t], columns=cols, index=pd.Index(["control", "treatment"], name="variant"))


def analyze(c, t, **kw):
    return analysis.analyze("x", "x", variants(c, t), **kw)


# --- stats ---


def test_two_proportion_test_reference():
    t = stats.two_proportion_test(100, 1000, 130, 1000)
    assert t["diff"] == pytest.approx(0.03) and t["lift"] == pytest.approx(0.3)
    se = math.sqrt(0.1 * 0.9 / 1000 + 0.13 * 0.87 / 1000)
    assert (t["ci_low"], t["ci_high"]) == pytest.approx((0.03 - 1.959964 * se, 0.03 + 1.959964 * se))
    assert t["z"] == pytest.approx(stats.two_proportion_z(130, 1000, 100, 1000))
    assert t["p_value"] == pytest.approx(0.0357, abs=1e-3) and t["significant"]
    assert not stats.two_proportion_test(100, 1000, 105, 1000)["significant"]


def test_mde_relative_inverts_sample_size():
    n = stats.sample_size_per_arm(0.10, 0.20)
    assert stats.mde_relative(0.10, n) == pytest.approx(0.20, abs=0.02)
    assert math.isinf(stats.mde_relative(0.0, 100))


# --- interpretation rules (one synthetic test per class) ---

BIG = 20000


@pytest.mark.parametrize("c,t,expected", [
    # clear positive: more leads and more SQLs, quality steady
    ((BIG, 600, 300, 120, 60, 20, 2e5, 1e4), (BIG, 800, 400, 160, 80, 27, 2.7e5, 1e4), "clear_positive"),
    # leads rise but SQL rate falls → volume-quality tradeoff
    ((BIG, 600, 300, 120, 60, 15, 1.5e5, 8e3), (BIG, 900, 430, 95, 50, 12, 1.2e5, 8e3), "volume_quality_tradeoff"),
    # more leads, downstream unchanged
    ((BIG, 600, 300, 120, 60, 20, 2e5, 1e4), (BIG, 700, 350, 125, 62, 20, 2e5, 1e4),
     "top_funnel_positive_downstream_neutral"),
    # fewer leads
    ((BIG, 600, 300, 120, 60, 20, 2e5, 1e4), (BIG, 480, 240, 96, 48, 16, 1.6e5, 1e4), "negative"),
    # primary flat, SQLs down
    ((BIG, 600, 300, 150, 60, 20, 2e5, 1e4), (BIG, 600, 300, 100, 50, 18, 1.8e5, 1e4), "negative"),
    # nothing moves
    ((BIG, 600, 300, 120, 60, 20, 2e5, 1e4), (BIG, 610, 305, 121, 60, 20, 2e5, 1e4), "inconclusive"),
])
def test_interpretation_rules(c, t, expected):
    assert analyze(c, t).interpretation == expected


def test_needs_longer_observation_when_window_shorter_than_velocity():
    c, t = (BIG, 600, 300, 120, 60, 20, 2e5, 1e4), (BIG, 700, 350, 125, 62, 20, 2e5, 1e4)
    short = analyze(c, t, velocity_days=60, start=date(2026, 7, 1), observed_through=date(2026, 8, 1))
    assert short.window_days == 31 and short.needs_longer
    assert short.interpretation == "needs_longer_observation" and short.recommendation == "observe_longer"
    long = analyze(c, t, velocity_days=60, start=date(2026, 3, 1), observed_through=date(2026, 8, 1))
    assert long.interpretation == "top_funnel_positive_downstream_neutral"
    # a clearly negative primary is reported as negative even if the window is short
    neg = analyze(c, (BIG, 480, 240, 96, 48, 16, 1.6e5, 1e4), velocity_days=60, start=date(2026, 7, 1),
                  observed_through=date(2026, 8, 1))
    assert neg.interpretation == "negative"


def test_recommendations():
    rec = {e: r for e, r in analysis.RECOMMENDATION_RULES.items()}
    assert rec == {"clear_positive": "scale", "top_funnel_positive_downstream_neutral": "iterate",
                   "volume_quality_tradeoff": "iterate", "negative": "stop",
                   "needs_longer_observation": "observe_longer"}
    small = analyze((1000, 50, 25, 10, 5, 2, 2e4, 1e3), (1000, 55, 27, 11, 5, 2, 2e4, 1e3))
    assert small.interpretation == "inconclusive" and small.recommendation == "retest"
    flat = analyze((200000, 6000, 3000, 1200, 600, 200, 2e6, 1e5), (200000, 6010, 3000, 1200, 600, 200, 2e6, 1e5))
    assert flat.interpretation == "inconclusive" and flat.recommendation == "stop"


def test_underpowered_flag_and_money():
    r = analyze((BIG, 600, 300, 120, 60, 20, 2e5, 1e4), (BIG, 800, 400, 160, 80, 0, 0, 1e4))
    assert r.stage("wins").underpowered and not r.stage("sqls").underpowered
    assert r.money.rpv_control == pytest.approx(10.0) and r.money.cac_control == pytest.approx(500)
    assert math.isinf(r.money.cac_treatment) and math.isnan(r.money.cac_change)
    assert analysis.analyze("x", "x", variants((1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1)),
                            settings=Settings(min_events=1)).stage("wins").events_per_arm == 1


def test_requires_two_variants():
    with pytest.raises(ValueError):
        analysis.analyze("x", "x", variants((1,) * 8, (1,) * 8).iloc[:1])


def test_velocity_for_mix():
    cells = pd.DataFrame({"source": ["email", "email"], "segment": ["smb", "mid_market"], "visitors": [300, 100]})
    vel = pd.DataFrame({"source": ["email", "email"], "segment": ["smb", "mid_market"],
                        "median_days_lead_to_win": [50, 100]})
    assert analysis.velocity_for_mix(cells, vel) == pytest.approx(62.5)
    assert analysis.velocity_for_mix(None, vel) is None


def test_interaction_z_detects_heterogeneity():
    seg = stats.two_proportion_test(300, 1000, 400, 1000)
    rest = stats.two_proportion_test(300, 1000, 300, 1000)
    assert analysis.interaction_z(seg, rest) > 3
    assert abs(analysis.interaction_z(seg, seg)) < 1e-9


# --- acceptance on the synthetic data ---


def test_every_interpretation_class_appears_in_past_experiments():
    classes = {e: run(e).interpretation for e in experiment_ids()}
    assert classes == {"EXP-01": "clear_positive", "EXP-02": "volume_quality_tradeoff", "EXP-03": "inconclusive",
                       "EXP-04": "top_funnel_positive_downstream_neutral", "EXP-05": "negative",
                       "EXP-06": "needs_longer_observation"}


def test_acceptance_webinar_short_form_is_volume_quality_tradeoff():
    r = run("EXP-02")
    assert r.stage("conversions").test["diff"] > 0 and r.stage("conversions").test["significant"]
    mql_sql = next(q for q in r.quality if q.label == "MQL→SQL")
    assert mql_sql.test["diff"] < 0 and mql_sql.test["significant"]
    assert r.interpretation == "volume_quality_tradeoff" and r.recommendation == "iterate"


def test_acceptance_manual_example_is_volume_quality_tradeoff():
    ex = VOLUME_QUALITY_EXAMPLE
    r = pipeline.run_manual(ex["name"], ex["variants"], ex["start"], ex["observed_through"], 60)
    assert r.interpretation == "volume_quality_tradeoff"


def test_segment_breakdown_finds_device_difference():
    r = run("EXP-04")
    assert any(f.dimension == "device" and f.value == "desktop" and f.differs for f in r.segments)
    assert run("EXP-02").segment_notes == []


def test_markdown_matches_spec_format():
    md = to_markdown(run("EXP-02"))
    assert [l for l in md.splitlines() if l.startswith("### ")] == [
        "### Immediate Result", "### Downstream Result", "### Quality Tradeoff", "### Segment Differences",
        "### Business Interpretation", "### Recommendation"]
    assert "**Volume-quality tradeoff**" in md and "**Iterate**" in md
    assert "⚠ Underpowered: Win" in md


def test_save_all_to_store(store):
    ids = analyze_all(store)
    assert ids == experiment_ids()
    e = store.get(Experiment, "EXP-06")
    assert e.interpretation == "needs_longer_observation" and e.impact_recommendation == "observe_longer"
    assert e.page == "nurture email" and e.variants[0].spend > 0 and e.statistical_result


def test_page_renders_csv_and_manual():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(config.ROOT / "views" / "8_Downstream_Impact_Analyzer.py")).run(timeout=60)
    assert not at.exception
    at.selectbox(key="dia_pick").select("EXP-02").run(timeout=60)
    assert at.metric[0].value == "Volume-quality tradeoff"
    at.button(key="dia_save").click().run(timeout=60)
    assert Store().get(Experiment, "EXP-02").interpretation == "volume_quality_tradeoff"
    at.radio(key="dia_source").set_value("Manual entry").run(timeout=60)
    assert not at.exception and at.metric[0].value == "Volume-quality tradeoff"
