from datetime import date

import pytest
from pydantic import ValidationError

from gios.core.schemas import (
    Experiment,
    Hypothesis,
    Learning,
    Opportunity,
    Recommendation,
    Signal,
    Variant,
)


def make_signal(**overrides):
    data = dict(
        type="behavioral",
        evidence_status="observed",
        source="web_behavior.csv",
        segment="smb",
        journey_stage="evaluation",
        summary="Pricing page exit rate for paid search is 1.6x baseline.",
        strength=4,
    )
    data.update(overrides)
    return Signal(**data)


def make_hypothesis(**scores):
    return Hypothesis(
        observed_problem="High pricing-page exits from paid search",
        affected_audience="SMB paid-search visitors",
        causal_explanation="Plan inclusions are unclear",
        intervention="Add a plan comparison table",
        expected_behavior_change="Fewer exits, more demo CTA clicks",
        expected_business_outcome="Lower paid CAC",
        **scores,
    )


# --- Signal ---


def test_signal_valid_and_has_id():
    s = make_signal()
    assert s.type == "behavioral"
    assert len(s.id) == 12
    assert s.id != make_signal().id


@pytest.mark.parametrize(
    "field,value",
    [
        ("type", "social"),
        ("evidence_status", "maybe"),
        ("segment", "enterprise"),
        ("journey_stage", "nowhere"),
        ("strength", 0),
        ("strength", 6),
        ("summary", ""),
        ("source", ""),
    ],
)
def test_signal_rejects_invalid(field, value):
    with pytest.raises(ValidationError):
        make_signal(**{field: value})


def test_signal_rejects_extra_fields():
    with pytest.raises(ValidationError):
        make_signal(confidence_pct=80)


def test_signal_all_types_accepted():
    for t in ["customer", "behavioral", "funnel", "acquisition", "operational", "revenue"]:
        assert make_signal(type=t).type == t


def test_signal_round_trip_json():
    s = make_signal()
    assert Signal.model_validate_json(s.model_dump_json()) == s


# --- Opportunity (GROWTH) ---


def test_opportunity_priority_score_formula():
    o = Opportunity(
        title="Clarify pricing", category="conversion",
        growth_impact=4, research_evidence=4, opportunity_size=3,
        web_evidence=5, technical_effort=2, hypothesis_confidence=3,
    )
    assert o.is_scored
    assert o.priority_score == (4 * 4 * 3 * 5 * 3) / 2


def test_opportunity_unscored_has_no_priority():
    o = Opportunity(title="x", category="acquisition", growth_impact=5)
    assert not o.is_scored
    assert o.priority_score is None


def test_opportunity_score_bounds():
    with pytest.raises(ValidationError):
        Opportunity(title="x", category="conversion", technical_effort=0)
    with pytest.raises(ValidationError):
        Opportunity(title="x", category="conversion", growth_impact=6)


def test_opportunity_priority_not_serialized():
    o = Opportunity(
        title="x", category="conversion", growth_impact=1, research_evidence=1,
        opportunity_size=1, web_evidence=1, technical_effort=1, hypothesis_confidence=1,
    )
    dumped = o.model_dump()
    assert "priority_score" not in dumped
    assert Opportunity.model_validate(dumped).priority_score == 1.0


# --- Hypothesis validation ---


@pytest.mark.parametrize(
    "scores,total,verdict",
    [
        ((0, 0, 0, 0, 0, 0), 0, "do_not_test"),
        ((1, 1, 1, 1, 0, 0), 4, "do_not_test"),
        ((1, 1, 1, 1, 1, 0), 5, "research_or_instrument_first"),
        ((2, 2, 1, 1, 1, 1), 8, "research_or_instrument_first"),
        ((2, 2, 2, 1, 1, 1), 9, "test_ready"),
        ((2, 2, 2, 2, 2, 2), 12, "test_ready"),
    ],
)
def test_hypothesis_verdict_thresholds(scores, total, verdict):
    names = [
        "evidence_diversity", "behavioral_support", "customer_support",
        "business_relevance", "testability", "measurement_readiness",
    ]
    h = make_hypothesis(**dict(zip(names, scores)))
    assert h.validation_total == total
    assert h.verdict == verdict


def test_hypothesis_unscored_and_bounds():
    assert make_hypothesis().verdict is None
    with pytest.raises(ValidationError):
        make_hypothesis(testability=3)


# --- Experiment ---


def test_experiment_valid():
    e = Experiment(
        name="Pricing table",
        variants=[
            Variant(name="control", visitors=1000, conversions=50, mqls=20, sqls=8, opps=4, wins=1),
            Variant(name="treatment", visitors=1000, conversions=65, mqls=25, sqls=9, opps=5, wins=2),
        ],
        start_date=date(2026, 3, 1),
        end_date=date(2026, 3, 28),
    )
    assert e.variants[1].conversion_rate == pytest.approx(0.065)


def test_experiment_requires_two_variants():
    with pytest.raises(ValidationError):
        Experiment(name="x", variants=[Variant(name="a", visitors=1, conversions=1)])


def test_experiment_rejects_bad_dates():
    with pytest.raises(ValidationError):
        Experiment(
            name="x",
            variants=[Variant(name="a", visitors=1, conversions=0), Variant(name="b", visitors=1, conversions=0)],
            start_date=date(2026, 3, 2),
            end_date=date(2026, 3, 1),
        )


def test_variant_rejects_non_monotonic_funnel():
    with pytest.raises(ValidationError):
        Variant(name="a", visitors=100, conversions=10, mqls=20)


# --- Learning / Recommendation ---


def test_learning_decision_vocab():
    assert Learning(what_happened="Lift held", decision="scale").decision == "scale"
    with pytest.raises(ValidationError):
        Learning(what_happened="x", decision="celebrate")


def test_recommendation_defaults_and_vocab():
    r = Recommendation(title="Fix mobile form", action_type="instrumentation_fix", rationale="Load time")
    assert r.horizon == "now" and r.confidence == "medium"
    with pytest.raises(ValidationError):
        Recommendation(title="x", action_type="vibes", rationale="y")


def test_signal_channel_period_and_overlap():
    s = make_signal(channel="paid_search", period="2026-03..2026-08")
    assert s.covers_month_range("2026-08", "2026-12")
    assert not s.covers_month_range("2026-09", "2026-12")
    assert not s.covers_month_range("2025-01", "2026-02")
    assert make_signal().covers_month_range("2020-01", "2020-02")  # unknown period: keep


def test_hypothesis_missing_parts_and_label():
    h = Hypothesis(observed_problem="Exits are high", intervention="  ")
    assert h.missing_parts == ["affected_audience", "causal_explanation", "intervention",
                               "expected_behavior_change", "expected_business_outcome"]
    assert h.label == "Exits are high"
    assert Hypothesis(title="T").label == "T"
