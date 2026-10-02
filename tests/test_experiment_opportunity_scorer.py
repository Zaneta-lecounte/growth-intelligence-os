import math

import pandas as pd
import pytest
from pydantic import ValidationError

from gios import config
from gios.core import stats
from gios.core.schemas import Hypothesis, Opportunity, Signal
from gios.core.store import Store
from gios.modules.demo_flow import seed_through
from gios.modules.experiment_opportunity_scorer import Settings, analysis, build_backlog, save, score
from gios.modules.experiment_opportunity_scorer.report import to_markdown


def sig(sid, type_, strength=3, summary="s"):
    return Signal(id=sid, type=type_, evidence_status="observed", source="x", segment="all",
                  journey_stage="evaluation", summary=summary, strength=strength)


def opp(**kw):
    base = dict(title="t", category="conversion", growth_impact=3, research_evidence=3, opportunity_size=3,
                web_evidence=3, technical_effort=3, hypothesis_confidence=3, validator_total=10)
    return Opportunity(**(base | kw))


# --- formula ---


def test_priority_formula_range():
    lo = opp(**{g: 1 for g in analysis.GROWTH} | {"technical_effort": 5})
    hi = opp(**{g: 5 for g in analysis.GROWTH} | {"technical_effort": 1})
    assert lo.priority_score == 0.2 and hi.priority_score == 3125
    assert opp(growth_impact=4, research_evidence=5, opportunity_size=2, web_evidence=3, technical_effort=2,
               hypothesis_confidence=5).priority_score == 4 * 5 * 2 * 3 * 5 / 2


# --- H mapping and prefills ---


def test_h_from_validator_mapping_aligned_with_bands():
    assert [analysis.h_from_validator(t) for t in range(13)] == [1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5]
    with pytest.raises(ValueError):
        analysis.h_from_validator(13)


def test_prefill_rules():
    linked = [sig("css-a", "customer", 5), sig("css-b", "customer", 2), sig("bfa-demo-device-mobile", "behavioral", 4),
              sig("qdl-leak-webinar-mql-to-sql", "funnel", 5)]
    h = Hypothesis(evidence_diversity=2, behavioral_support=2, customer_support=2, business_relevance=1,
                   testability=1, measurement_readiness=1)  # total 9
    scores, notes = analysis.prefill(h, linked)
    assert scores == {"growth_impact": 3, "research_evidence": 5, "opportunity_size": 5, "web_evidence": 4,
                      "technical_effort": 3, "hypothesis_confidence": 4}
    assert "validator total 9/12" in notes["hypothesis_confidence"]
    funnel_only, _ = analysis.prefill(Hypothesis(confidence="low"), [sig("qdl-flag-x-y", "operational")])
    assert (funnel_only["research_evidence"], funnel_only["web_evidence"], funnel_only["opportunity_size"],
            funnel_only["hypothesis_confidence"]) == (1, 2, 3, 2)
    nothing, notes = analysis.prefill(None, [])
    assert nothing["web_evidence"] == 1 and "not validated" in notes["hypothesis_confidence"]


# --- id parsing, population, category ---


def test_parse_signal_ids():
    assert analysis.parse_friction_id("bfa-pricing-source-paid_search") == {"page": "pricing", "channel": "paid_search"}
    assert analysis.parse_friction_id("bfa-demo-device-mobile") == {"page": "demo", "device": "mobile"}
    assert analysis.parse_leak_id("qdl-leak-paid_search-lead-to-mql") == {"channel": "paid_search",
                                                                         "funnel_stage": "Lead→MQL"}
    assert analysis.parse_flag_id("qdl-flag-qualification_mismatch-webinar") == {"channel": "webinar",
                                                                                "funnel_stage": "MQL→SQL"}
    assert analysis.parse_flag_id("qdl-flag-routing_loss-all") is None
    assert analysis.parse_leak_id("css-x") is None


def test_infer_population_prefers_web_slice():
    assert analysis.infer_population(["qdl-leak-webinar-mql-to-sql", "bfa-demo-device-mobile"]) == \
        {"test_unit": "web", "page": "demo", "device": "mobile"}
    assert analysis.infer_population(["css-x", "qdl-flag-sla_loss-partner"])["funnel_stage"] == "MQL→SQL"
    assert analysis.infer_population(["css-x"]) == {}


def test_category_and_fix_prefill():
    assert analysis.category_for([sig("qdl-leak-webinar-mql-to-sql", "funnel")]) == "qualification"
    assert analysis.category_for([sig("qdl-flag-sla_loss-partner", "operational")]) == "operations_measurement"
    assert analysis.category_for([sig("bfa-demo-device-mobile", "behavioral")]) == "conversion"
    assert analysis.category_for([sig("qdl-flag-rising_cac-paid_search", "acquisition"),
                                  sig("qdl-leak-paid_search-lead-to-mql", "funnel")]) == "acquisition"
    assert analysis.category_for([sig("css-a", "customer")]) == "customer_problem"
    assert analysis.is_fix([sig("b", "behavioral", summary="… Friction: technical (unconfirmed)")])
    assert not analysis.is_fix([sig("b", "behavioral", summary="Friction: comprehension")])


# --- sample size ---


def test_sample_size_reference_values():
    assert stats.sample_size_per_arm(0.10, 0.20) == 3841        # 10% -> 12%, α .05, power .8
    assert stats.sample_size_per_arm(0.05, 0.10) == 31234       # 5% -> 5.5%
    assert stats.sample_size_per_arm(0.10, 0.40) < stats.sample_size_per_arm(0.10, 0.20)
    assert stats.sample_size_per_arm(0.0, 0.2) == 0 and stats.sample_size_per_arm(0.9, 0.2) == 0
    assert stats.weeks_to_sample(1000, 500) == 4.0 and math.isinf(stats.weeks_to_sample(1, 0))


def _web(sessions_per_day, completes_per_day):
    dates = pd.date_range("2026-08-01", "2026-08-31").strftime("%Y-%m-%d")
    return pd.DataFrame({"date": dates, "page": "demo", "source": "organic", "device": "mobile",
                         "sessions": sessions_per_day, "cta_clicks": 0, "form_completes": completes_per_day})


def test_sample_check_flags_when_traffic_cannot_reach_sample_in_six_weeks():
    o = opp(test_unit="web", page="demo", device="mobile", mde=0.2)
    busy = analysis.sample_check(o, _web(2000, 200), pd.DataFrame(), Settings())
    assert busy.baseline == pytest.approx(0.10) and busy.weekly_volume == pytest.approx(14000)
    assert busy.n_per_arm == 3841 and not busy.flagged
    quiet = analysis.sample_check(o, _web(100, 10), pd.DataFrame(), Settings())
    assert quiet.flagged and quiet.weeks == pytest.approx(2 * 3841 / 700)
    bigger_effect = analysis.sample_check(o.model_copy(update={"mde": 1.0}), _web(100, 10), pd.DataFrame(), Settings())
    assert not bigger_effect.flagged
    none = analysis.sample_check(opp(), _web(1, 1), pd.DataFrame(), Settings())
    assert not none.available and not none.flagged


def test_sample_check_funnel_population():
    funnel = pd.DataFrame([{"month": "2026-08", "source": "webinar", "segment": "smb", "visits": 0, "leads": 0,
                            "mqls": 1300, "sqls": 143, "opps": 0, "wins": 0}])
    c = analysis.sample_check(opp(test_unit="funnel", channel="webinar", funnel_stage="MQL→SQL"),
                              pd.DataFrame({"date": ["2026-08-31"]}), funnel, Settings())
    assert c.baseline == pytest.approx(0.11) and c.weekly_volume == pytest.approx(1300 * 12 / 52)


# --- recommendation rules ---


@pytest.mark.parametrize("kw,expected", [
    (dict(validator_total=4), "reject"),
    (dict(validator_total=8, growth_impact=5, research_evidence=5, opportunity_size=5, web_evidence=5,
          hypothesis_confidence=5, technical_effort=1), "research_first"),           # regardless of score
    (dict(validator_total=6, risk_flags=["measurement_readiness"]), "instrument_first"),
    (dict(validator_total=None), "research_first"),
    (dict(risk_flags=["measurement_readiness"]), "instrument_first"),
    (dict(risk_flags=["legal_privacy"]), "research_first"),
    (dict(growth_impact=1, research_evidence=1), "defer"),                           # 3*3*3/3 = 9 < 20
    (dict(risk_flags=["sample_size", "brand"]), "run_now"),
])
def test_recommendation_rules(kw, expected):
    assert analysis.recommend(opp(**kw), Settings())[0] == expected


def test_validator_instrument_first_is_respected():
    assert analysis.recommend(opp(validator_total=7), Settings(), "instrument_first")[0] == "instrument_first"


def test_override_requires_reason():
    with pytest.raises(ValidationError):
        opp(recommendation_override="run_now")
    with pytest.raises(ValidationError):
        opp(recommendation_override="run_now", override_reason="ok")
    o = opp(recommendation="research_first", recommendation_override="run_now",
            override_reason="Board mandate to ship before Q4")
    assert o.final_recommendation == "run_now"


# --- integration ---


@pytest.fixture(scope="module")
def seeded(tmp_path_factory):
    store = Store(tmp_path_factory.mktemp("eos") / "s.db")
    seed_through("validations", store)
    return store


def test_backlog_from_hypotheses_and_acceptance(seeded):
    sb = score(build_backlog(seeded), Settings(), seeded)
    by_hyp = {o.hypothesis_id: o for o in sb.items}
    assert len(by_hyp) == 7
    pricing = by_hyp["gid-paid_search-1"]
    assert (pricing.hypothesis_confidence, pricing.research_evidence, pricing.web_evidence) == (5, 5, 4)
    assert pricing.final_recommendation == "run_now" and pricing.page == "pricing"
    webinar = by_hyp["gid-all-1"]
    assert "sample_size" in webinar.risk_flags and webinar.final_recommendation == "run_now"
    assert by_hyp["gid-paid_search-3"].title.startswith("Rising bids")
    assert by_hyp["gid-paid_search-3"].final_recommendation == "research_first"   # validator 5
    mobile = by_hyp["gid-paid_search-2"]
    assert mobile.is_fix and mobile.category == "conversion"
    assert sb.ranked()[0].priority_score == 400


def test_backlog_keeps_edits_and_follows_validator(seeded, tmp_path):
    store = Store(tmp_path / "copy.db")
    store.save_many(seeded.list(Signal))
    store.save_many(seeded.list(Hypothesis))
    backlog = build_backlog(store)
    o = next(x for x in backlog if x.hypothesis_id == "gid-all-1")
    o.growth_impact, o.technical_effort = 5, 2
    manual = opp(id="eos-manual-x", title="Manual idea", validator_total=None)
    save(backlog + [manual], store)
    h = store.get(Hypothesis, "gid-all-1")
    h.evidence_diversity = h.behavioral_support = h.customer_support = 2
    h.business_relevance = h.testability = h.measurement_readiness = 2
    store.save(h)
    again = {x.id: x for x in build_backlog(store)}
    assert again[o.id].growth_impact == 5 and again[o.id].technical_effort == 2
    assert again[o.id].validator_total == 12 and again[o.id].hypothesis_confidence == 5
    assert "eos-manual-x" in again
    save([again["eos-manual-x"]], store)
    assert [x.id for x in store.list(Opportunity)] == ["eos-manual-x"]


def test_markdown_matches_spec_format(seeded):
    md = to_markdown(score(build_backlog(seeded), Settings(), seeded))
    assert "| Opportunity | G | R | O | W | T | H | Score | Key Risk |" in md
    assert [l for l in md.splitlines() if l.startswith("### ")] == ["### Recommendation", "### Rationale"]
    for label in ["Run now", "Research first", "Instrument first", "Defer", "Reject"]:
        assert f"- **{label}**:" in md
    assert "does not replace it" in md


def test_page_seeds_scores_and_saves():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(config.ROOT / "views" / "6_Experiment_Opportunity_Scorer.py")).run(timeout=60)
    at.button(key="eos_seed").click().run(timeout=180)
    assert not at.exception
    assert any("does not replace it" in i.value for i in at.info)
    ranked = next(d.value for d in at.dataframe if "Score" in d.value.columns)
    assert ranked.Score.iloc[0] == 400
    at.button(key="eos_save").click().run(timeout=60)
    assert len(Store().list(Opportunity)) == 7
