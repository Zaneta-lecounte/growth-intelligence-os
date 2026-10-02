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
    for name in ["funnel_by_source", "web_behavior", "customer_evidence", "sales_feedback", "experiments",
                 "experiment_segments", "lead_velocity"]:
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
        "experiment_segments": gen.EXPERIMENT_SEGMENT_COLUMNS,
        "lead_velocity": gen.VELOCITY_COLUMNS,
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
                        mqls=r.mqls, sqls=r.sqls, opps=r.opps, wins=r.wins, revenue=r.revenue, spend=r.spend)
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


# --- Helpers ----------------------------------------------------------------------------------------

HIGH_INTENT = {"consideration", "evaluation", "purchase"}


def _rising(series: pd.Series, min_up_steps: int = 3) -> bool:
    """Trends upward without being cartoonishly monotonic: positive slope, most steps up."""
    import numpy as np

    y = series.to_numpy(dtype=float)
    slope = np.polyfit(np.arange(len(y)), y, 1)[0]
    return slope > 0 and int((np.diff(y) > 0).sum()) >= min_up_steps


def _stage_rates(funnel: pd.DataFrame) -> pd.DataFrame:
    t = funnel.groupby("source")[["visits", "leads", "mqls", "sqls", "opps", "wins"]].sum()
    return pd.DataFrame({"leads": t.leads, "lead_to_mql": t.mqls / t.leads, "mql_to_sql": t.sqls / t.mqls,
                         "sql_to_opp": t.opps / t.sqls, "wins": t.wins})


# --- Story 1: paid search -> pricing page ----------------------------------------------------


def test_story1_paid_search_is_a_meaningful_source(funnel):
    share = funnel.groupby("source").visits.sum() / funnel.visits.sum()
    assert share["paid_search"] > 0.2


def test_story1_pricing_clarity_verbatims_rising(evidence):
    pricing = evidence[evidence._theme == "pricing_clarity"]
    monthly = pricing.groupby("month").size()
    assert _rising(monthly)
    assert monthly.iloc[-2:].sum() > 2.5 * monthly.iloc[:2].sum()
    assert not monthly.is_monotonic_increasing  # realistic: at least one month dips
    # Detectable from text alone, not only the hidden tag
    raw = evidence[evidence.verbatim.str.contains(r"pricing|plan|included|includes|price|cost", case=False)]
    raw_monthly = raw.groupby("month").size()
    assert raw_monthly.iloc[-1] > 2 * raw_monthly.iloc[0]


def test_story1_pricing_verbatims_are_high_intent_and_cover_each_confusion(evidence):
    pricing = evidence[evidence._theme == "pricing_clarity"]
    assert pricing.journey_stage.isin(HIGH_INTENT).mean() >= 0.95
    text = pricing.verbatim.str.lower()
    kinds = {
        "structure": r"seats and minutes|asterisks|pricing felt hidden|tell the plans apart",
        "inclusions": r"what's included|what's actually in|includes|add-on|capped|excluded",
        "plan fit": r"which plan fits|difference between|look almost the same",
        "sales before cost": r"talk to sales|book a call|demo before|ballpark",
    }
    for kind, pattern in kinds.items():
        assert text.str.contains(pattern).sum() >= 3, kind


def test_story1_paid_search_pricing_exits_high_and_rising(web):
    pricing = web[web.page == "pricing"]
    ps = pricing[pricing.source == "paid_search"].groupby("month")[["exits", "sessions"]].sum()
    other = pricing[pricing.source != "paid_search"].groupby("month")[["exits", "sessions"]].sum()
    ps_rate, other_rate = ps.exits / ps.sessions, other.exits / other.sessions
    assert _rising(ps_rate, min_up_steps=4)
    assert ps_rate.iloc[-1] - ps_rate.iloc[0] > 0.15
    assert (ps_rate - other_rate > 0.03).all() and (ps_rate.iloc[-3:] - other_rate.iloc[-3:] > 0.15).all()
    assert other_rate.max() - other_rate.min() < 0.03  # other sources stable
    assert ps_rate.diff().iloc[1:].min() < 0.03  # one softer month, not a straight line
    # Paid search traffic exits other pages at a normal rate: the friction is on pricing.
    ps_elsewhere = web[(web.page != "pricing") & (web.source == "paid_search")]
    all_elsewhere = web[web.page != "pricing"]
    assert abs(ps_elsewhere.exits.sum() / ps_elsewhere.sessions.sum()
               - all_elsewhere.exits.sum() / all_elsewhere.sessions.sum()) < 0.03


def test_story1_paid_search_cac_rising(funnel):
    ps_ratio = _cac_ratio(funnel, "paid_search")
    assert ps_ratio > 1.3
    for source in gen.SOURCES:
        if source != "paid_search":
            assert _cac_ratio(funnel, source) < ps_ratio
    monthly = funnel[funnel.source == "paid_search"].groupby("month")[["spend", "wins"]].sum()
    assert _rising(monthly.spend / monthly.wins)


# --- Story 2: webinar qualification mismatch ---------------------------------------------------


def test_story2_webinar_volume_and_lead_to_mql_strong_but_mql_to_sql_weak(funnel):
    r = _stage_rates(funnel)
    assert r.leads.idxmax() == "webinar"
    assert r.lead_to_mql.idxmax() == "webinar"
    assert r.mql_to_sql.idxmin() == "webinar"
    assert r.mql_to_sql["webinar"] < 0.5 * r.mql_to_sql.drop("webinar").median()


def test_story2_leak_is_at_mql_to_sql_not_later(funnel):
    r = _stage_rates(funnel)
    others = funnel[funnel.source != "webinar"][["sqls", "opps"]].sum()
    assert abs(r.sql_to_opp["webinar"] - others.opps / others.sqls) < 0.05  # reasonable once at SQL
    assert r.wins["webinar"] >= 0.1 * r.wins.sum()  # still a real source of revenue


def test_story2_rejections_are_qualification_not_follow_up_speed(data):
    sales = data["sales_feedback"]
    rejected = sales[sales.rejection_reason.fillna("") != ""]
    qual_share = rejected.rejection_reason.isin(gen.QUALIFICATION_REASONS).groupby(rejected.source).mean()
    assert qual_share["webinar"] > 0.75
    assert qual_share["webinar"] > qual_share.drop("webinar").max() + 0.2
    webinar_reasons = set(rejected[rejected.source == "webinar"].rejection_reason)
    assert {"low_intent", "educational_only", "company_too_small", "not_in_market",
            "student_or_researcher"} <= webinar_reasons
    # Follow-up speed is NOT the cause: webinar is no slower than the median source.
    hours = sales.groupby("source").hours_to_first_follow_up.median()
    assert hours["webinar"] <= hours.median() * 1.1


# --- Story 3: mobile demo form technical friction ----------------------------------------------


def _demo(web):
    return web[web.page == "demo"].groupby("device")[
        ["sessions", "exits", "form_starts", "form_completes", "rage_clicks"]].sum()


def test_story3_mobile_demo_form(web):
    demo = _demo(web)
    start_rate = demo.form_starts / demo.sessions
    completion = demo.form_completes / demo.form_starts
    assert demo.form_starts["mobile"] > 20_000          # healthy mobile intent
    assert start_rate["mobile"] >= start_rate["desktop"]  # not a discoverability problem
    assert completion["mobile"] < 0.5 * completion["desktop"]
    rage = demo.rage_clicks / demo.sessions
    assert rage["mobile"] > 3 * rage["desktop"]


def test_story3_mobile_demo_exits_elevated_and_strongest_on_demo(web):
    rates = web.groupby(["page", "device"])[["exits", "sessions"]].sum()
    exit_rate = (rates.exits / rates.sessions).unstack()
    gap = exit_rate.mobile - exit_rate.desktop
    assert gap["demo"] > 0.05
    assert gap.drop("demo").max() < 0.05 and (gap.drop("demo") > 0).all()  # normal device gap elsewhere
    assert gap.idxmax() == "demo"


def test_story3_mobile_demo_page_slow(web):
    load = web.groupby([web.page == "demo", "device"]).avg_load_ms.mean()
    assert load[(True, "mobile")] > 2 * load[(True, "desktop")]
    assert load[(True, "mobile")] > 1.8 * load[(False, "mobile")]
    assert load[(True, "mobile")] > 4000
    other = web[web.page != "demo"].groupby("device").avg_load_ms.mean()
    assert 1.2 < other["mobile"] / other["desktop"] < 1.9  # normal mobile slowdown elsewhere


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


# --- Realistic noise ---------------------------------------------------------------------------


def test_noise_event_months_and_mix_drift(funnel):
    visits = funnel.groupby(["source", "month"]).visits.sum().unstack()
    assert visits.loc["webinar", "2026-05"] > 1.2 * visits.loc["webinar"].median()   # flagship webinar
    neighbours = visits.loc["paid_social", ["2026-05", "2026-07"]].mean()
    assert visits.loc["paid_social", "2026-06"] < 0.85 * neighbours  # campaign paused
    import numpy as np

    share = visits / visits.sum()
    months = np.arange(share.shape[1])
    social = share.loc["paid_social"].drop("2026-06")  # trend outside the paused month
    assert np.polyfit(months[[0, 1, 2, 4, 5]], social.to_numpy(), 1)[0] > 0   # growing share
    assert np.polyfit(months, share.loc["email"].to_numpy(), 1)[0] < 0         # shrinking share


def test_noise_segment_differences(data, funnel):
    seg = funnel.groupby("segment")[["mqls", "sqls"]].sum()
    assert (seg.sqls / seg.mqls)["mid_market"] > (seg.sqls / seg.mqls)["smb"]
    hours = data["sales_feedback"].groupby("routed_to").hours_to_first_follow_up.median()
    assert hours["mid_market_ae"] < hours["smb_sdr"] < hours["partner_team"] < hours["unassigned"]


def test_noise_some_sources_strong_top_funnel_neutral_downstream(funnel):
    t = funnel.groupby("source")[["visits", "leads", "sqls", "wins"]].sum()
    visit_to_lead = t.leads / t.visits
    lead_to_win = t.wins / t.leads
    email = "email"
    assert visit_to_lead[email] > visit_to_lead.drop(["webinar", email]).median()
    assert abs(lead_to_win[email] - lead_to_win.drop("webinar").median()) < 0.01


def test_noise_outlier_experiment(data):
    exp = data["experiments"].set_index(["experiment_id", "variant"])
    o = exp.loc["EXP-07"]
    lift = (o.conversions / o.visitors).iloc[1] / (o.conversions / o.visitors).iloc[0] - 1
    assert lift > 1.0 and o.visitors.max() < 1000
    assert (pd.to_datetime(o.observed_through.iloc[0]) - pd.to_datetime(o.start_date.iloc[0])).days < 30


# --- Experiments: cells, totals, velocity -------------------------------------------------------


def test_experiment_cells_sum_to_variant_totals(data):
    cells, totals = data["experiment_segments"], data["experiments"]
    cols = ["visitors", "conversions", "mqls", "sqls", "opps", "wins", "revenue", "spend"]
    summed = cells.groupby(["experiment_id", "variant"])[cols].sum().reset_index()
    merged = totals.merge(summed, on=["experiment_id", "variant"], suffixes=("", "_cells"))
    for c in cols:
        assert (merged[c] == merged[f"{c}_cells"]).all(), c
    assert set(cells.device) == {"desktop", "mobile"} and set(cells.segment) == {"smb", "mid_market"}
    assert (totals.observed_through >= totals.end_date).all()


def test_lead_velocity(data, funnel):
    v = data["lead_velocity"]
    assert len(v) == 12
    assert (v.p25_days < v.median_days_lead_to_win).all() and (v.median_days_lead_to_win < v.p75_days).all()
    mid, smb = v[v.segment == "mid_market"].median_days_lead_to_win, v[v.segment == "smb"].median_days_lead_to_win
    assert mid.min() > smb.max()
    assert v.won_deals.sum() == funnel.wins.sum()


def test_experiment_generation_keeps_other_streams():
    """Experiments are generated last, so editing them never shifts the stories above."""
    import numpy as np

    rng = np.random.default_rng(gen.SEED)
    funnel = gen.generate_funnel(rng)
    pd.testing.assert_frame_equal(funnel, gen.generate_all(gen.SEED)["funnel_by_source"])
