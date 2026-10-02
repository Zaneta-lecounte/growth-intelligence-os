"""Tests for scripts/generate_data.py: reproducibility, integrity, and the embedded stories."""
import pandas as pd
import pytest

from gios import config
from scripts import generate_data as gen


@pytest.fixture(scope="module")
def data():
    return gen.generate_all(gen.SEED)


@pytest.fixture(scope="module")
def funnel(data):
    return data["funnel_by_source"]


@pytest.fixture(scope="module")
def web(data):
    w = data["web_behavior"].copy()
    w["month"] = w["date"].str[:7]
    return w


@pytest.fixture(scope="module")
def evidence(data):
    e = data["customer_evidence"].copy()
    e["month"] = e["date"].str[:7]
    return e


def _cac_ratio(funnel: pd.DataFrame, source: str) -> float:
    """Spend/wins over the last 3 months divided by the first 3 months."""
    f = funnel[funnel.source == source]
    months = sorted(f.month.unique())
    early, late = f[f.month.isin(months[:3])], f[f.month.isin(months[3:])]
    return (late.spend.sum() / late.wins.sum()) / (early.spend.sum() / early.wins.sum())


# --- Reproducibility --------------------------------------------------------------------------


def test_same_seed_is_identical():
    a, b = gen.generate_all(7), gen.generate_all(7)
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name])


def test_different_seed_differs():
    a, b = gen.generate_all(1), gen.generate_all(2)
    assert not a["funnel_by_source"].equals(b["funnel_by_source"])


def test_committed_csvs_match_generator(tmp_path):
    gen.write_all(tmp_path)
    for name in ["funnel_by_source", "web_behavior", "customer_evidence", "sales_feedback", "experiments"]:
        committed = config.DATA_DIR / f"{name}.csv"
        assert committed.exists(), f"run scripts/generate_data.py ({name}.csv missing)"
        pd.testing.assert_frame_equal(pd.read_csv(committed), pd.read_csv(tmp_path / f"{name}.csv"))


# --- Shape and integrity ----------------------------------------------------------------------


def test_columns_match_spec(tmp_path):
    paths = gen.write_all(tmp_path)
    expected = {
        "funnel_by_source": gen.FUNNEL_COLUMNS,
        "web_behavior": gen.WEB_COLUMNS,
        "customer_evidence": gen.EVIDENCE_COLUMNS,  # untagged: no theme column
        "sales_feedback": gen.SALES_COLUMNS,
        "experiments": gen.EXPERIMENT_COLUMNS,
    }
    for name, cols in expected.items():
        assert list(pd.read_csv(paths[name], nrows=1).columns) == cols


def test_funnel_covers_six_months_all_sources_segments(funnel):
    assert funnel.month.nunique() == 6
    assert set(funnel.source) == set(gen.SOURCES)
    assert set(funnel.segment) == {"smb", "mid_market"}
    assert len(funnel) == 6 * 6 * 2


def test_funnel_is_monotonic_and_non_negative(funnel):
    stages = ["visits", "leads", "mqls", "sqls", "opps", "wins"]
    for upper, lower in zip(stages, stages[1:]):
        assert (funnel[upper] >= funnel[lower]).all(), f"{upper} < {lower}"
    assert (funnel[stages + ["revenue", "spend"]] >= 0).all().all()
    assert ((funnel.wins > 0) == (funnel.revenue > 0)).all()


def test_web_behavior_integrity(web):
    assert set(web.page) == set(gen.PAGES)
    assert set(web.device) == {"desktop", "mobile"}
    assert (web.exits <= web.sessions).all()
    assert (web.scroll_75 <= web.sessions - web.exits).all()
    assert (web.cta_clicks <= web.sessions).all()
    assert (web.form_starts <= web.sessions).all()
    assert (web.form_completes <= web.form_starts).all()
    assert (web.avg_load_ms > 0).all()
    assert web.date.min() == "2026-03-01" and web.date.max() == "2026-08-31"


def test_sales_feedback_matches_funnel(data, funnel):
    sales = data["sales_feedback"]
    assert sales.lead_id.is_unique
    per_source = sales.groupby("source").size()
    accepted = sales[sales.rejection_reason.isna() | (sales.rejection_reason == "")].groupby("source").size()
    mqls = funnel.groupby("source").mqls.sum()
    sqls = funnel.groupby("source").sqls.sum()
    pd.testing.assert_series_equal(per_source.sort_index(), mqls.sort_index(), check_names=False)
    pd.testing.assert_series_equal(accepted.sort_index(), sqls.sort_index(), check_names=False)
    assert (sales.hours_to_first_follow_up > 0).all()


def test_experiments_valid_against_schema(data):
    from gios.core.schemas import Experiment, Variant

    exps = data["experiments"]
    assert exps.experiment_id.nunique() >= 5
    for exp_id, g in exps.groupby("experiment_id"):
        exp = Experiment(
            name=g.experiment_name.iloc[0],
            hypothesis_text=g.hypothesis.iloc[0],
            primary_metric=g.primary_metric.iloc[0],
            start_date=g.start_date.iloc[0],
            end_date=g.end_date.iloc[0],
            status="completed",
            variants=[
                Variant(name=r.variant, visitors=r.visitors, conversions=r.conversions,
                        mqls=r.mqls, sqls=r.sqls, opps=r.opps, wins=r.wins)
                for r in g.itertuples()
            ],
        )
        assert len(exp.variants) == 2


def test_evidence_uses_schema_vocab(evidence):
    from typing import get_args

    from gios.core.schemas import JourneyStage

    assert set(evidence.source) <= set(gen.EVIDENCE_SOURCES)
    assert set(evidence.segment) <= {"smb", "mid_market"}
    assert set(evidence.journey_stage) <= set(get_args(JourneyStage))
    assert evidence.verbatim.str.len().min() > 10


# --- Story 1: paid search -> pricing page ----------------------------------------------------


def test_story1_pricing_clarity_verbatims_rising(evidence):
    monthly = evidence[evidence._theme == "pricing_clarity"].groupby("month").size()
    assert monthly.is_monotonic_increasing
    assert monthly.iloc[-2:].sum() > 2.5 * monthly.iloc[:2].sum()
    # Detectable from text alone, not only the hidden tag
    raw = evidence[evidence.verbatim.str.contains(r"pricing|plan|included|includes", case=False)]
    raw_monthly = raw.groupby("month").size()
    assert raw_monthly.iloc[-1] > 2 * raw_monthly.iloc[0]


def test_story1_paid_search_pricing_exits_high_and_rising(web):
    pricing = web[web.page == "pricing"]
    ps = pricing[pricing.source == "paid_search"].groupby("month")[["exits", "sessions"]].sum()
    other = pricing[pricing.source != "paid_search"].groupby("month")[["exits", "sessions"]].sum()
    ps_rate, other_rate = ps.exits / ps.sessions, other.exits / other.sessions
    assert ps_rate.is_monotonic_increasing
    assert ps_rate.iloc[-1] - ps_rate.iloc[0] > 0.15
    assert (ps_rate.iloc[-3:] - other_rate.iloc[-3:] > 0.15).all()
    assert other_rate.max() - other_rate.min() < 0.03  # other sources stable


def test_story1_paid_search_cac_rising(funnel):
    ps_ratio = _cac_ratio(funnel, "paid_search")
    assert ps_ratio > 1.4
    for source in gen.SOURCES:
        if source != "paid_search":
            assert _cac_ratio(funnel, source) < ps_ratio


# --- Story 2: webinar qualification mismatch ---------------------------------------------------


def test_story2_webinar_volume_and_lead_to_mql_strong_but_mql_to_sql_weak(funnel):
    by_source = funnel.groupby("source")[["leads", "mqls", "sqls"]].sum()
    lead_to_mql = by_source.mqls / by_source.leads
    mql_to_sql = by_source.sqls / by_source.mqls
    assert by_source.leads.idxmax() == "webinar"
    assert lead_to_mql.idxmax() == "webinar"
    assert mql_to_sql.idxmin() == "webinar"
    assert mql_to_sql["webinar"] < 0.5 * mql_to_sql.drop("webinar").median()


def test_story2_rejections_are_qualification_not_follow_up_speed(data):
    sales = data["sales_feedback"]
    rejected = sales[sales.rejection_reason.fillna("") != ""]
    qual_share = rejected.rejection_reason.isin(gen.QUALIFICATION_REASONS).groupby(rejected.source).mean()
    assert qual_share["webinar"] > 0.75
    assert qual_share["webinar"] > qual_share.drop("webinar").max() + 0.2
    assert "student_or_researcher" in set(rejected[rejected.source == "webinar"].rejection_reason)
    # Follow-up speed is NOT the cause: webinar is no slower than the median source.
    hours = sales.groupby("source").hours_to_first_follow_up.median()
    assert hours["webinar"] <= hours.median() * 1.1


# --- Story 3: mobile demo form technical friction ----------------------------------------------


def test_story3_mobile_demo_form(web):
    demo = web[web.page == "demo"].groupby("device")[["sessions", "form_starts", "form_completes", "rage_clicks"]].sum()
    start_rate = demo.form_starts / demo.sessions
    completion = demo.form_completes / demo.form_starts
    assert start_rate["mobile"] >= start_rate["desktop"]
    assert completion["mobile"] < 0.5 * completion["desktop"]
    rage = demo.rage_clicks / demo.sessions
    assert rage["mobile"] > 3 * rage["desktop"]


def test_story3_mobile_demo_page_slow(web):
    load = web.groupby([web.page == "demo", "device"]).avg_load_ms.mean()
    assert load[(True, "mobile")] > 2 * load[(True, "desktop")]
    assert load[(True, "mobile")] > 1.8 * load[(False, "mobile")]
    assert load[(True, "mobile")] > 4000


def test_story3_customer_evidence_mentions_mobile_demo(evidence):
    mentions = evidence.verbatim.str.contains(r"phone|mobile|iPhone|Android", case=False) & \
        evidence.verbatim.str.contains("demo", case=False)
    assert mentions.sum() >= 20


# --- Red herring: frequent but low-severity complaint -------------------------------------------


def test_red_herring_is_most_frequent_but_post_purchase(evidence):
    counts = evidence._theme.value_counts()
    assert counts.idxmax() == "dark_mode_red_herring"
    assert counts["dark_mode_red_herring"] > counts["pricing_clarity"]
    herring = evidence[evidence._theme == "dark_mode_red_herring"]
    # Low severity: never raised in evaluation/purchase, flat over time, never in win/loss.
    assert set(herring.journey_stage) <= {"onboarding", "retention"}
    assert "win_loss" not in set(herring.source)
    monthly = herring.groupby("month").size()
    assert monthly.max() - monthly.min() <= 5


# --- Robustness: the stories are structural, not an artifact of one seed ------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_stories_hold_across_seeds(seed):
    d = gen.generate_all(seed)
    w = d["web_behavior"].assign(month=lambda x: x.date.str[:7])
    e = d["customer_evidence"].assign(month=lambda x: x.date.str[:7])
    f = d["funnel_by_source"]
    test_story1_pricing_clarity_verbatims_rising(e)
    test_story1_paid_search_pricing_exits_high_and_rising(w)
    test_story1_paid_search_cac_rising(f)
    test_story2_webinar_volume_and_lead_to_mql_strong_but_mql_to_sql_weak(f)
    test_story2_rejections_are_qualification_not_follow_up_speed(d)
    test_story3_mobile_demo_form(w)
    test_story3_mobile_demo_page_slow(w)
    test_red_herring_is_most_frequent_but_post_purchase(e)
