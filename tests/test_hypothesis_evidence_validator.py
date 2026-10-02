import json

import pytest
from pydantic import ValidationError

from gios import config
from gios.core.schemas import Hypothesis, Signal
from gios.core.store import Store
from gios.modules.hypothesis_evidence_validator import analysis, pipeline, save, validate
from gios.modules.hypothesis_evidence_validator.examples import ASSUMPTION_ONLY
from gios.modules.hypothesis_evidence_validator.models import DIMENSIONS, DimensionScore, ValidatorProposal
from gios.modules.hypothesis_evidence_validator.report import to_markdown
from gios.modules.signal_layer import run_signal_layer
from tests.conftest import FakeClient


def sig(sid, type_):
    return Signal(id=sid, type=type_, evidence_status="observed", source="x", segment="all",
                  journey_stage="evaluation", summary=sid, strength=3)


SIGNALS = [sig("c", "customer"), sig("b", "behavioral"), sig("f", "funnel"), sig("o", "operational")]


def hyp(signal_ids=(), **kw):
    base = {p: "a reasonably specific statement" for p in
            ["observed_problem", "affected_audience", "causal_explanation", "intervention",
             "expected_behavior_change", "expected_business_outcome"]}
    return Hypothesis(signal_ids=list(signal_ids), **(base | kw))


def proposal(score=2):
    return ValidatorProposal(scores=[DimensionScore(dimension=d, score=score, justification="j") for d in DIMENSIONS])


# --- standard ---


def test_standard_flags_missing_and_vague_parts():
    checks = {c.part: c.status for c in analysis.check_standard(hyp(intervention="", affected_audience="SMBs"))}
    assert checks["intervention"] == "missing" and checks["affected_audience"] == "vague"
    assert checks["observed_problem"] == "ok"


# --- evidence map and caps ---


def test_evidence_map_groups_by_type_and_tracks_unknown_ids():
    h = hyp(["c", "b", "ghost", "c"], contradicting_signal_ids=["o"])
    em = analysis.evidence_map(h, SIGNALS)
    assert em.supporting_types == {"customer", "behavioral"}
    assert [s.id for s in em.contradicting["operational"]] == ["o"]
    assert em.unknown_ids == ["ghost"] and em.n_supporting == 2


@pytest.mark.parametrize("ids,expected", [
    ((), {"evidence_diversity": 0, "behavioral_support": 0, "customer_support": 0, "business_relevance": 1,
          "testability": 2, "measurement_readiness": 1}),
    (("c",), {"evidence_diversity": 1, "behavioral_support": 0, "customer_support": 2, "business_relevance": 1,
              "testability": 2, "measurement_readiness": 1}),
    (("c", "b", "f"), {d: 2 for d in DIMENSIONS}),
    (("o",), {"evidence_diversity": 1, "behavioral_support": 0, "customer_support": 0, "business_relevance": 1,
              "testability": 2, "measurement_readiness": 1}),
])
def test_evidence_caps(ids, expected):
    h = hyp(ids)
    caps = analysis.evidence_caps(analysis.evidence_map(h, SIGNALS), analysis.check_standard(h))
    assert {d: c[0] for d, c in caps.items()} == expected


def test_testability_cap_from_missing_parts():
    for kw, cap in (({"intervention": ""}, 1), ({"intervention": "", "expected_behavior_change": "x"}, 0)):
        h = hyp(("c", "b", "f"), **kw)
        caps = analysis.evidence_caps(analysis.evidence_map(h, SIGNALS), analysis.check_standard(h))
        assert caps["testability"][0] == cap


# --- scoring, band, recommendation ---


def test_scores_capped_then_overridden():
    h = hyp()
    caps = analysis.evidence_caps(analysis.evidence_map(h, SIGNALS), analysis.check_standard(h))
    rows = analysis.score_rows(proposal(2), caps)
    assert analysis.total(rows) == 4 and analysis.band(4) == "do_not_test"
    assert next(r for r in rows if r.dimension == "business_relevance").capped
    rows = analysis.score_rows(proposal(2), caps, overrides={"customer_support": 2})
    r = next(r for r in rows if r.dimension == "customer_support")
    assert r.score == 2 and r.overridden and r.above_cap
    with pytest.raises(ValueError):
        analysis.score_rows(proposal(2), caps, overrides={"testability": 3})
    assert analysis.total(analysis.score_rows(None, caps)) == 0  # no proposal: manual scoring


@pytest.mark.parametrize("total,expected", [(0, "do_not_test"), (4, "do_not_test"), (5, "research_or_instrument_first"),
                                            (8, "research_or_instrument_first"), (9, "test_ready"), (12, "test_ready")])
def test_band_thresholds_match_schema(total, expected):
    assert analysis.band(total) == expected
    scores = [2] * (total // 2) + [1] * (total % 2)
    scores += [0] * (6 - len(scores))
    assert Hypothesis(**dict(zip(DIMENSIONS, scores))).verdict == expected


def _rows(**scores):
    return [analysis.DimensionRow(d, None, 2, "", scores.get(d, 0), "") for d in DIMENSIONS]


def test_recommendation_rules():
    assert analysis.recommendation(_rows(**{d: 2 for d in DIMENSIONS})) == "test"
    assert analysis.recommendation(_rows(evidence_diversity=1, behavioral_support=1, customer_support=1,
                                         business_relevance=2, testability=1, measurement_readiness=1)) == "instrument_first"
    assert analysis.recommendation(_rows(evidence_diversity=1, behavioral_support=0, customer_support=0,
                                         business_relevance=2, testability=1, measurement_readiness=1)) == "research_first"
    assert analysis.recommendation(_rows(business_relevance=2, testability=2)) == "reject"


def test_weakest_area_tie_breaks_in_spec_order():
    assert analysis.weakest(_rows(evidence_diversity=1, behavioral_support=0, customer_support=0)).dimension == \
        "behavioral_support"


def test_proposal_must_cover_each_dimension_once():
    with pytest.raises(ValidationError):
        ValidatorProposal(scores=proposal().scores[:5])
    with pytest.raises(ValidationError):
        ValidatorProposal(scores=proposal().scores[:5] + proposal().scores[:1])


# --- LLM path ---


def test_live_proposal_is_capped_and_retried(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    bad = proposal().model_dump() | {"scores": proposal().model_dump()["scores"][:4]}
    client = FakeClient([json.dumps(bad), proposal(2).model_dump_json()])
    result = validate(hyp(), SIGNALS, client=client)
    assert len(client.calls) == 2
    assert "No signals are linked to this hypothesis" in client.calls[0]["messages"][0]["content"]
    assert result.total == 4 and result.recommendation == "reject"


def test_missing_demo_cache_falls_back_to_manual_scoring():
    result = validate(hyp(), SIGNALS)
    assert result.proposal is None and "no cached output" in result.proposal_error
    assert result.total == 0


# --- acceptance and integration ---


@pytest.fixture(scope="module")
def phase1(tmp_path_factory):
    store = Store(tmp_path_factory.mktemp("hev") / "s.db")
    run_signal_layer(store)
    return store.list(Signal)


def test_acceptance_assumption_only_scores_at_most_four(phase1):
    result = validate(ASSUMPTION_ONLY, phase1)
    assert result.proposal is not None
    assert result.total <= 4 and result.band == "do_not_test" and result.recommendation == "reject"
    # caps hold even if the LLM proposed full marks
    generous = validate(ASSUMPTION_ONLY, phase1, proposal=proposal(2))
    assert generous.total <= 4


def test_diagnostic_top_hypothesis_is_test_ready(phase1):
    from gios.modules.growth_intelligence_diagnostic import BusinessSignal, Filters
    from gios.modules.growth_intelligence_diagnostic import pipeline as gid

    business = BusinessSignal(goal="Grow pipeline", target_metric="CAC", what="Paid search CAC rose",
                              baseline="Earlier half", segment="all", period="2026-03 to 2026-08")
    top = gid.to_hypotheses(gid.run(business, Filters(channel="paid_search"), phase1))[0]
    result = validate(top, phase1)
    assert result.total >= 9 and result.recommendation == "test"
    assert result.evidence.supporting_types >= {"customer", "behavioral", "acquisition"}


def test_markdown_matches_spec_format(phase1):
    md = to_markdown(validate(ASSUMPTION_ONLY, phase1))
    assert [l for l in md.splitlines() if l.startswith("### ")] == [
        "### Hypothesis", "### Evidence Map", "### Score", "### Weakest Evidence Area", "### Missing Evidence",
        "### Recommendation"]
    assert "**Total: 4 / 12**" in md and "**Reject**" in md and "capped from 2" in md


def test_save_writes_scores_and_keeps_module(phase1, store):
    h = hyp(["css-pricing_uncertainty"])
    store.save(h, module="growth_intelligence_diagnostic")
    result = validate(h, phase1, proposal=proposal(1))
    save(result, store)
    saved = store.get(Hypothesis, h.id)
    assert saved.validation_total == result.total and saved.verdict == result.band
    assert saved.validator_recommendation == result.recommendation
    assert set(saved.score_justifications) == set(DIMENSIONS)
    assert store.module_of(Hypothesis, h.id) == "growth_intelligence_diagnostic"
    new = ASSUMPTION_ONLY.model_copy()
    save(validate(new, phase1), store)
    assert store.module_of(Hypothesis, new.id) == "hypothesis_evidence_validator"


def test_demo_cache_files_validate():
    files = list(config.DEMO_DIR.glob("validate_hypothesis__*.json"))
    assert len(files) >= 6
    for f in files:
        ValidatorProposal.model_validate_json(f.read_text())


def test_page_manual_example_scores_reject_and_saves():
    from streamlit.testing.v1 import AppTest

    run_signal_layer(Store())
    at = AppTest.from_file(str(config.ROOT / "views" / "5_Hypothesis_Evidence_Validator.py")).run(timeout=60)
    assert not at.exception
    assert any("**Total: 4 / 12**" in m.value for m in at.markdown)
    assert at.metric[2].value == "Reject"
    at.button(key="hev_save").click().run(timeout=60)
    assert Store().get(Hypothesis, ASSUMPTION_ONLY.id).validator_recommendation == "reject"


def test_page_stored_hypothesis(monkeypatch):
    from streamlit.testing.v1 import AppTest

    from gios.modules.growth_intelligence_diagnostic import BusinessSignal, Filters
    from gios.modules.growth_intelligence_diagnostic import pipeline as gid

    store = Store()
    run_signal_layer(store)
    business = BusinessSignal(goal="Grow pipeline", target_metric="CAC", what="Paid search CAC rose",
                              baseline="Earlier half", segment="all", period="2026-03 to 2026-08")
    gid.save(gid.run(business, Filters(channel="paid_search"), store.list(Signal)), store)
    at = AppTest.from_file(str(config.ROOT / "views" / "5_Hypothesis_Evidence_Validator.py")).run(timeout=60)
    assert at.radio(key="hev_mode").value == "Stored hypothesis"
    at.selectbox(key="hev_pick").select("gid-paid_search-1").run(timeout=60)
    assert not at.exception
    assert at.metric[2].value == "Test"
