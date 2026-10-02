import re
import json

import pytest

from gios import config
from gios.core.schemas import Opportunity
from gios.core.store import Store
from gios.modules.demo_flow import seed_through
from gios.modules.growth_priority_orchestrator import analysis, build, gather, propose, rebalance, save
from gios.modules.growth_priority_orchestrator.models import OrchestratorProposal
from gios.modules.growth_priority_orchestrator.pipeline import ProposalResult
from gios.modules.growth_priority_orchestrator.report import to_markdown
from tests.conftest import FakeClient


def opp(oid, score_h=4, bucket=None, horizon=None, category="conversion", rec="run_now", **kw):
    base = dict(id=oid, title=oid, category=category, growth_impact=3, research_evidence=3, opportunity_size=3,
                web_evidence=3, technical_effort=3, hypothesis_confidence=score_h, recommendation=rec,
                portfolio_bucket=bucket, horizon=horizon)
    return Opportunity(**(base | kw))


# --- merging ---


def test_overlap():
    a, b = opp("a", signal_ids=["x", "y"]), opp("b", signal_ids=["y", "z"])
    assert analysis.overlap(a, b) == pytest.approx(1 / 3)
    assert analysis.cluster_overlap([a, b, opp("c")]) == 0


def test_merge_keeps_highest_priority_as_primary():
    a = opp("a", score_h=2, signal_ids=["x"], dependencies=["data"])
    b = opp("b", score_h=5, signal_ids=["y"])
    merged = {o.id: o for o in analysis.merge([a, b, opp("c")], [(["a", "b"], "Canonical")])}
    assert merged["a"].merged_into == "b" and merged["b"].merged_into is None
    assert merged["b"].title == "Canonical" and merged["b"].merged_ids == ["a"]
    assert merged["b"].signal_ids == ["y", "x"] and merged["b"].dependencies == ["data"]
    assert len(analysis.active(merged.values())) == 2
    # a cluster that no longer has two unmerged members is skipped
    again = analysis.merge(list(merged.values()), [(["a", "b"], None), (["c", "ghost"], None)])
    assert {o.id: o.merged_into for o in again} == {"a": "b", "b": None, "c": None}


# --- classification defaults ---


@pytest.mark.parametrize("kw,bucket", [
    (dict(rec="research_first"), "research_bet"),
    (dict(score_h=2), "research_bet"),
    (dict(is_fix=True), "foundational_fix"),
    (dict(rec="instrument_first"), "foundational_fix"),
    (dict(category="operations_measurement"), "foundational_fix"),
    (dict(score_h=4), "high_confidence_optimization"),
    (dict(score_h=4, technical_effort=4), "strategic_experiment"),
    (dict(score_h=3), "strategic_experiment"),
])
def test_default_bucket(kw, bucket):
    assert analysis.default_bucket(opp("x", **kw)) == bucket


def test_rule_dependencies_and_horizon():
    assert analysis.rule_dependencies(opp("x", rec="instrument_first")) == ["data"]
    assert analysis.rule_dependencies(opp("x", is_fix=True, test_unit="web")) == ["engineering"]
    assert analysis.rule_dependencies(opp("x", test_unit="web")) == ["design", "content"]
    assert analysis.rule_dependencies(opp("x", category="qualification")) == ["sales", "operations"]
    assert [analysis.default_horizon(opp("x", rec=r)) for r in
            ["run_now", "instrument_first", "research_first", "defer", "reject"]] == ["now", "now", "next", "later", None]
    o = opp("x", rec="run_now", recommendation_override="defer", override_reason="Waiting on Q1 budget")
    assert analysis.default_horizon(o) == "later"


def test_assign_defaults_orders_by_score_and_respects_user_horizon():
    items = analysis.assign_defaults([opp("low", score_h=3), opp("high", score_h=5),
                                      opp("pinned", horizon="later", horizon_order=1)])
    by = {o.id: o for o in items}
    assert (by["high"].horizon, by["high"].horizon_order) == ("now", 1)
    assert (by["low"].horizon, by["low"].horizon_order) == ("now", 2)
    assert by["pinned"].horizon == "later"


# --- balance ---


def test_balance_shares_and_bucket_warning():
    items = [opp(f"f{i}", bucket="foundational_fix", horizon="now") for i in range(3)] + \
            [opp("r", bucket="research_bet", horizon="next", category="qualification"),
             opp("merged", bucket="research_bet", horizon=None, merged_into="r"),
             opp("off", bucket="research_bet", horizon=None)]
    b = analysis.balance(items)
    assert b.counts == {"foundational_fix": 3, "high_confidence_optimization": 0, "strategic_experiment": 0,
                        "research_bet": 1}
    assert b.shares["foundational_fix"] == 0.75
    assert any("Foundational fixes make up 75%" in w for w in b.warnings)
    assert not any("top-funnel" in w for w in b.warnings)
    assert "No strategic experiments in the plan." in b.notes


def test_balance_exactly_half_is_fine_and_all_top_funnel_warns():
    items = [opp("a", bucket="research_bet", horizon="now", category="acquisition"),
             opp("b", bucket="foundational_fix", horizon="next", category="conversion")]
    b = analysis.balance(items)
    assert not any("make up" in w for w in b.warnings)
    assert any("top-funnel" in w for w in b.warnings)


def test_balance_capacity_and_empty():
    items = [opp(f"n{i}", bucket=list(analysis.BUCKETS)[i % 4], horizon="now", category="qualification")
             for i in range(4)]
    assert any("capacity is 3" in w for w in analysis.balance(items, now_capacity=3).warnings)
    assert analysis.balance(items, now_capacity=4).warnings == []
    assert analysis.balance([]).warnings == ["The roadmap is empty."]


# --- LLM proposal filtering ---


def test_propose_drops_unknown_ids(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    raw = {"clusters": [{"member_ids": ["a", "ghost"], "canonical_title": "x", "rationale": "y"},
                        {"member_ids": ["a", "b", "ghost"], "canonical_title": "x", "rationale": "y"}],
           "items": [{"item_id": "ghost", "dependencies": [], "expected_learning": "l", "primary_metric": "m"},
                     {"item_id": "a", "dependencies": ["data"], "expected_learning": "l", "primary_metric": "m"}]}
    result = propose([opp("a"), opp("b")], {}, client=FakeClient([json.dumps(raw)]))
    assert result.ignored_ids == ["ghost"]
    assert [c.member_ids for c in result.proposal.clusters] == [["a", "b"]]
    assert list(result.tags) == ["a"]


# --- end to end on the demo store ---


@pytest.fixture(scope="module")
def seeded(tmp_path_factory):
    store = Store(tmp_path_factory.mktemp("gpo") / "s.db")
    seed_through("backlog", store)
    return store


@pytest.fixture(scope="module")
def roadmap(seeded):
    inputs = gather(seeded)
    proposal = propose(inputs.candidates, inputs.hypotheses)
    merges = [(c.member_ids, c.canonical_title) for c in proposal.proposal.clusters]
    return build(inputs, proposal, merges, now_capacity=3)


def test_demo_roadmap(roadmap):
    assert len(roadmap.items) == 7 and len(roadmap.active) == 5
    assert [o.title for o in roadmap.lane("now")] == ["Clarify pricing for paid search evaluators",
                                                      "Fix the mobile demo form", "Webinar leads are learners, not buyers"]
    assert {o.title for o in roadmap.lane("next")} == {"Slow partner follow-up loses partner demand",
                                                       "Rising bids are buying lower-intent paid search traffic"}
    assert roadmap.balance.warnings == []
    buckets = {o.title: o.portfolio_bucket for o in roadmap.active}
    assert buckets["Fix the mobile demo form"] == "foundational_fix"
    assert buckets["Clarify pricing for paid search evaluators"] == "high_confidence_optimization"


def test_markdown_matches_spec_format(roadmap):
    md = to_markdown(roadmap)
    assert [l for l in md.splitlines() if l.startswith("### ")] == [
        "### Executive Growth Priorities", "### Now", "### Next", "### Later",
        "### Research / Instrumentation Required", "### Key Dependencies", "### Expected Learning", "### Metrics"]
    assert "1. **Clarify pricing for paid search evaluators**" in md
    assert re.search(r"Webinar leads are learners, not buyers\*\*: the sample takes ~\d+ weeks", md)  # sample-size risk


def test_user_edits_rebalance(roadmap):
    import copy

    edited = copy.deepcopy(roadmap)
    for o in edited.items:
        if not o.merged_into:
            o.portfolio_bucket = "strategic_experiment"
    edited = rebalance(edited)
    assert any("Strategic experiments make up 100%" in w for w in edited.balance.warnings)


def test_save_and_reload(seeded, roadmap):
    save(roadmap, seeded)
    stored = {o.id: o for o in seeded.list(Opportunity)}
    # tie on score and title -> the lower id is the primary
    assert stored["eos-gid-paid_search-1"].merged_into == "eos-gid-all-2"
    assert stored["eos-gid-all-2"].horizon == "now" and stored["eos-gid-paid_search-1"].horizon is None
    again = gather(seeded)
    assert sum(1 for o in again.candidates if o.merged_into) == 2


def test_demo_cache_matches_demo_backlog(seeded):
    cached = OrchestratorProposal.model_validate_json((config.DEMO_DIR / "propose_roadmap_clusters.json").read_text())
    ids = {o.id for o in gather(seeded).candidates}
    assert {t.item_id for t in cached.items} == ids
    assert all(set(c.member_ids) <= ids for c in cached.clusters)


def test_page_confirms_merges_and_saves():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(config.ROOT / "views" / "7_Growth_Priority_Orchestrator.py")).run(timeout=60)
    at.button(key="gpo_seed").click().run(timeout=180)
    assert not at.exception
    assert not any("Merged with" in m.value for m in at.markdown)   # nothing merged until confirmed
    at.checkbox(key="gpo_merge_0").check().run(timeout=60)
    at.checkbox(key="gpo_merge_1").check().run(timeout=60)
    assert not at.exception
    assert any("5 roadmap items after merging duplicates (2 merged)" in m.value for m in at.markdown)
    at.button(key="gpo_save").click().run(timeout=60)
    assert sum(1 for o in Store().list(Opportunity) if o.merged_into) == 2
