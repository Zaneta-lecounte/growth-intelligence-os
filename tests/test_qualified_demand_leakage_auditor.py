import pandas as pd
import pytest

from gios import config
from gios.core import stats
from gios.core.schemas import Signal
from gios.core.store import Store
from gios.modules.qualified_demand_leakage_auditor import Settings, analysis, pipeline, run, to_signals
from gios.modules.qualified_demand_leakage_auditor.models import LeakageNarrative
from gios.modules.qualified_demand_leakage_auditor.report import to_markdown
from tests.conftest import FakeClient


def funnel(rows):
    cols = ["source", "segment", "visits", "leads", "mqls", "sqls", "opps", "wins", "revenue", "spend"]
    df = pd.DataFrame(rows, columns=cols)
    df.insert(0, "month", "2026-03")
    return df


@pytest.fixture
def toy():
    return funnel([
        ["a", "smb", 10000, 500, 400, 40, 20, 2, 20000, 10000],     # strong top, weak MQL→SQL
        ["b", "smb", 10000, 300, 120, 50, 25, 6, 60000, 20000],
        ["c", "smb", 10000, 300, 120, 48, 24, 6, 60000, 20000],
        ["d", "smb", 100, 10, 5, 2, 1, 0, 0, 1000],                  # tiny: insufficient volume
    ])


def sales(rows):
    return pd.DataFrame(rows, columns=["lead_id", "source", "rejection_reason", "hours_to_first_follow_up", "routed_to"])


# --- stage map ---


def test_stage_map_rates_cis_and_benchmarks(toy):
    sm = analysis.stage_map(toy, ["source"], min_volume=30).set_index(["source", "stage"])
    r = sm.loc[("a", "MQL→SQL")]
    assert r.rate == pytest.approx(0.1)
    assert (r.ci_low, r.ci_high) == pytest.approx(stats.wilson_ci(40, 400))
    assert r.benchmark == pytest.approx((50 + 48 + 2) / (120 + 120 + 5))
    assert r.status == "below"
    assert sm.loc[("a", "Lead→MQL")].status == "above"
    assert sm.loc[("d", "MQL→SQL")].status == "insufficient volume"
    assert sm.loc[("a", "Lead→Win")].rate == pytest.approx(2 / 500)


def test_segment_benchmark_uses_same_segment_peers():
    f = funnel([
        ["a", "smb", 1000, 100, 50, 10, 5, 1, 1, 1],
        ["b", "smb", 1000, 100, 50, 25, 5, 1, 1, 1],
        ["b", "mid_market", 1000, 100, 50, 45, 5, 1, 1, 1],
    ])
    sm = analysis.stage_map(f, ["source", "segment"]).set_index(["source", "segment", "stage"])
    assert sm.loc[("a", "smb", "MQL→SQL")].benchmark == pytest.approx(0.5)


def test_economics(toy):
    e = analysis.economics(toy)
    assert e.loc["a", "cpl"] == pytest.approx(20)
    assert e.loc["a", "cac"] == pytest.approx(5000)
    assert e.loc["a", "revenue_per_win"] == pytest.approx(10000)
    assert pd.isna(e.loc["d", "cac"])


# --- follow-up and rejections ---


def test_follow_up_and_rejection_mix():
    s = sales([
        ["1", "a", "", 2, "smb_sdr"], ["2", "a", "no_budget", 30, "smb_sdr"],
        ["3", "a", "student_or_researcher", 3, "smb_sdr"], ["4", "a", "no_response", 50, "unassigned"],
        ["5", "b", "", 1, "smb_sdr"],
    ])
    fu = analysis.follow_up(s, Settings(sla_hours=24))
    assert fu.loc["a", "late_share"] == 0.5
    assert fu.loc["a", "accept_on_time"] == 0.5 and fu.loc["a", "accept_late"] == 0
    assert fu.loc["a", "unassigned"] == 1
    mix = analysis.rejection_mix(s)
    assert mix.loc["a", "rejections"] == 3
    assert mix.loc["a", "qualification_share"] == pytest.approx(2 / 3)


# --- flags ---


def test_flags_on_toy(toy):
    s = sales([[str(i), "a", "no_budget" if i % 10 else "", 3, "smb_sdr"] for i in range(400)]
              + [[f"b{i}", "b", "no_response" if i % 2 else "", 30 if i % 3 == 0 else 2, "smb_sdr"] for i in range(120)])
    a = analysis.analyze(toy, s, Settings(min_volume=30))
    kinds = {(f.kind, f.source) for f in a.flags}
    assert ("strong_top_weak_downstream", "a") in kinds
    assert ("qualification_mismatch", "a") in kinds
    assert ("high_volume_low_quality", "a") in kinds
    assert ("sla_loss", "b") in kinds
    assert not any(f.source == "d" for f in a.flags)
    assert ("qualification_mismatch", "b") not in kinds


def test_owner_for():
    q = [analysis.Flag("qualification_mismatch", "a", "qualification", "")]
    sla = [analysis.Flag("sla_loss", "a", "operations_routing", "")]
    assert analysis.owner_for("a", "MQL→SQL", q) == "qualification"
    assert analysis.owner_for("a", "MQL→SQL", sla) == "operations_routing"
    assert analysis.owner_for("a", "MQL→SQL", []) == "sales_handoff"
    assert analysis.owner_for("a", "Visit→Lead", []) == "web_conversion"
    assert analysis.owner_for("a", "Lead→MQL", []) == "acquisition"
    assert analysis.owner_for("a", "SQL→Opp", []) == "sales_handoff"
    assert analysis.owner_for("a", "Opp→Win", []) == "product_offering"
    assert analysis.owner_for("a", "Opp→Win", [], sufficient=False) == "measurement"


def test_sizing_propagates_through_own_downstream_rates(toy):
    a = analysis.analyze(toy, sales([]).astype({"hours_to_first_follow_up": float}), Settings(gap_closure=0.5))
    leak = a.leaks.set_index(["source", "stage"]).loc[("a", "MQL→SQL")]
    bench = (50 + 48 + 2) / 245
    extra = 400 * 0.5 * (bench - 0.1)
    assert leak.extra_conversions == pytest.approx(extra)
    assert leak.extra_wins == pytest.approx(extra * (20 / 40) * (2 / 20))
    assert leak.extra_revenue == pytest.approx(leak.extra_wins * 10000)
    assert leak.lifted_rate == pytest.approx(0.1 + 0.5 * (bench - 0.1))
    full = analysis.analyze(toy, sales([]).astype({"hours_to_first_follow_up": float}), Settings(gap_closure=1.0))
    assert full.leaks.set_index(["source", "stage"]).loc[("a", "MQL→SQL")].extra_wins == pytest.approx(2 * leak.extra_wins)


# --- narrative guard ---


def narrative(about):
    return LeakageNarrative(about_leak=about, leakage_summary="s", probable_causes=[], recommended_intervention="i",
                            primary_metric="p", downstream_metrics=[])


def test_narrative_rejected_when_about_another_leak(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    result = run(client=FakeClient([narrative("organic|Visit→Lead").model_dump_json()]))
    assert result.narrative is None and "not the current highest-value leak" in result.narrative_note
    assert "Narrative unavailable" in to_markdown(result)


def test_narrative_cannot_contain_numbers(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    bad = narrative("webinar|MQL→SQL").model_dump() | {"leakage_summary": "Webinar converts at 11%."}
    good = narrative("webinar|MQL→SQL").model_dump_json()
    import json

    client = FakeClient([json.dumps(bad), good])
    result = run(client=client)
    assert len(client.calls) == 2 and result.narrative is not None


# --- acceptance on synthetic data ---


@pytest.fixture(scope="module")
def demo():
    return run()


def test_acceptance_webinar_mql_to_sql_is_highest_value_leak(demo):
    top = demo.analysis.top_leak
    assert (top.source, top.stage) == ("webinar", "MQL→SQL")
    assert top.owner == "qualification"
    assert top.extra_revenue_per_month > 1.5 * demo.analysis.leaks.iloc[1].extra_revenue_per_month
    assert demo.narrative is not None


@pytest.mark.parametrize("gap", [0.1, 0.25, 0.5, 0.75, 1.0])
def test_acceptance_holds_for_any_gap_closure(gap):
    top = run(settings=Settings(gap_closure=gap)).analysis.top_leak
    assert (top.source, top.stage) == ("webinar", "MQL→SQL")


def test_acceptance_flags(demo):
    kinds = {(f.kind, f.source) for f in demo.analysis.flags}
    assert {("qualification_mismatch", "webinar"), ("strong_top_weak_downstream", "webinar"),
            ("high_volume_low_quality", "webinar"), ("sla_loss", "partner"), ("routing_loss", "all"),
            ("rising_cac", "paid_search")} == kinds
    # webinar follow-up speed is fine: not an SLA problem
    assert demo.analysis.follow_up.loc["webinar", "late_share"] < 0.05


def test_markdown_matches_spec_format(demo):
    md = to_markdown(demo)
    headings = [l for l in md.splitlines() if l.startswith("### ")]
    assert headings == ["### Leakage Summary", "### Highest-Value Leakage Point", "### Evidence",
                        "### Probable Cause Categories", "### Recommended Owner(s)", "### Recommended Intervention",
                        "### Primary / Downstream Metrics"]
    assert "**webinar · MQL→SQL**" in md
    assert "- **Qualification** (primary)" in md


def test_signals(demo, store):
    signals = to_signals(demo)
    by_id = {s.id: s for s in signals}
    assert by_id["qdl-leak-webinar-mql-to-sql"].strength == 5
    assert by_id["qdl-flag-sla_loss-partner"].type == "operational"
    assert by_id["qdl-flag-high_volume_low_quality-webinar"].type == "acquisition"
    assert len({s.id for s in signals}) == len(signals)
    store.replace_module_output(signals, "qualified_demand_leakage_auditor")


def test_page_renders_and_saves():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(config.ROOT / "views" / "3_Qualified_Demand_Leakage_Auditor.py")).run(timeout=60)
    assert not at.exception
    assert any("webinar · MQL→SQL" in m.value for m in at.markdown)
    at.button(key="qdl_save").click().run(timeout=60)
    assert len(Store().list(Signal, module="qualified_demand_leakage_auditor")) == len(to_signals(run()))


def test_cac_trend_and_rising_cac_flag():
    rows = []
    for i, m in enumerate(["2026-01", "2026-02", "2026-03", "2026-04"]):
        rows.append([m, "a", "smb", 1000, 100, 50, 40, 40, 40 if i < 2 else 20, 1, 4000])
        rows.append([m, "b", "smb", 1000, 100, 50, 40, 40, 40, 1, 4000])
    f = pd.DataFrame(rows, columns=["month", "source", "segment", "visits", "leads", "mqls", "sqls", "opps",
                                    "wins", "revenue", "spend"])
    cac = analysis.cac_trend(f)
    assert cac.loc["a", "early_cac"] == pytest.approx(100) and cac.loc["a", "late_cac"] == pytest.approx(200)
    assert cac.loc["a", "change"] == pytest.approx(1.0) and cac.loc["b", "change"] == 0
    a = analysis.analyze(f, sales([]).astype({"hours_to_first_follow_up": float}), Settings(min_volume=30))
    assert [(x.kind, x.source) for x in a.flags if x.kind == "rising_cac"] == [("rising_cac", "a")]
    few = analysis.analyze(f, sales([]).astype({"hours_to_first_follow_up": float}), Settings(min_volume=50))
    assert not [x for x in few.flags if x.kind == "rising_cac"]


def test_signals_carry_channel_and_period(demo):
    by_id = {s.id: s for s in to_signals(demo)}
    assert by_id["qdl-flag-rising_cac-paid_search"].channel == "paid_search"
    assert by_id["qdl-flag-rising_cac-paid_search"].type == "acquisition"
    assert by_id["qdl-flag-routing_loss-all"].channel is None
    assert all(s.period == "2026-03..2026-08" for s in by_id.values())
