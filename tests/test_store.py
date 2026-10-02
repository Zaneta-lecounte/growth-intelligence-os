import pytest

from gios.core.schemas import (
    Experiment,
    Hypothesis,
    Learning,
    Opportunity,
    Recommendation,
    Signal,
    Variant,
)
from gios.core.store import Store


def sig(summary="Paid search pricing exits up", **kw):
    data = dict(
        type="behavioral", evidence_status="observed", source="web_behavior.csv",
        segment="smb", journey_stage="evaluation", summary=summary, strength=4,
    )
    data.update(kw)
    return Signal(**data)


def test_creates_db_file(tmp_path):
    path = tmp_path / "x.db"
    Store(path)
    assert path.exists()


def test_save_and_get_round_trip(store):
    s = sig()
    assert store.save(s, module="behavioral_friction") == s.id
    assert store.get(Signal, s.id) == s


def test_get_missing_returns_none(store):
    assert store.get(Signal, "nope") is None


def test_list_preserves_order_and_filters_by_module(store):
    a, b, c = sig("a"), sig("b"), sig("c")
    store.save(a, module="customer_signal")
    store.save_many([b, c], module="behavioral_friction")
    assert [s.summary for s in store.list(Signal)] == ["a", "b", "c"]
    assert [s.summary for s in store.list(Signal, module="behavioral_friction")] == ["b", "c"]


def test_save_is_upsert(store):
    s = sig()
    store.save(s)
    s.strength = 2
    store.save(s)
    assert len(store.list(Signal)) == 1
    assert store.get(Signal, s.id).strength == 2


def test_all_entity_types_round_trip(store):
    opp = Opportunity(
        title="Clarify pricing", category="conversion", growth_impact=4, research_evidence=4,
        opportunity_size=3, web_evidence=5, technical_effort=2, hypothesis_confidence=3,
        risk_flags=["sample_size"],
    )
    hyp = Hypothesis(
        opportunity_id=opp.id, observed_problem="p", affected_audience="a",
        causal_explanation="c", intervention="i", expected_behavior_change="b",
        expected_business_outcome="o", evidence_diversity=2,
    )
    exp = Experiment(
        name="e", hypothesis_id=hyp.id,
        variants=[Variant(name="a", visitors=10, conversions=1), Variant(name="b", visitors=10, conversions=2)],
    )
    learn = Learning(experiment_id=exp.id, what_happened="w", decision="iterate")
    rec = Recommendation(title="t", action_type="experiment", rationale="r", opportunity_id=opp.id)
    for obj in (opp, hyp, exp, learn, rec):
        store.save(obj)
        loaded = store.get(type(obj), obj.id)
        assert loaded == obj
    assert store.get(Opportunity, opp.id).priority_score == opp.priority_score


def test_modules_can_pass_outputs_across_store_instances(tmp_path):
    path = tmp_path / "shared.db"
    s = sig()
    Store(path).save(s, module="customer_signal")
    assert Store(path).list(Signal, module="customer_signal") == [s]


def test_delete_clear_counts(store):
    a, b = sig("a"), sig("b")
    store.save_many([a, b])
    store.save(Learning(what_happened="w", decision="stop"))
    assert store.counts()["signals"] == 2
    assert store.delete(Signal, a.id)
    assert not store.delete(Signal, a.id)
    store.clear(Signal)
    assert store.counts()["signals"] == 0
    assert store.counts()["learnings"] == 1
    store.clear()
    assert sum(store.counts().values()) == 0


def test_rejects_unknown_model(store):
    from gios.core.schemas import Variant

    with pytest.raises(TypeError):
        store.list(Variant)
