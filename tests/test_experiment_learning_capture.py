import pytest
from pydantic import ValidationError

from gios import config
from gios.core.schemas import Experiment, Hypothesis, Learning
from gios.core.store import Store
from gios.modules.downstream_impact_analyzer import analyze_all
from gios.modules.experiment_learning_capture import build, check, draft, draft_all, pipeline, prefill, save, \
    send_next_hypothesis
from gios.modules.experiment_learning_capture import library
from gios.modules.experiment_learning_capture.models import LearningDraft
from gios.modules.experiment_learning_capture.report import to_markdown


@pytest.fixture(scope="module")
def analyzed(tmp_path_factory):
    store = Store(tmp_path_factory.mktemp("elc") / "s.db")
    analyze_all(store)
    return store


def exp(store, eid="EXP-02"):
    return store.get(Experiment, eid)


def test_prefill_is_deterministic_from_impact_result(analyzed):
    p = prefill(exp(analyzed))
    assert p["decision"] == "iterate" and p["interpretation"] == "volume_quality_tradeoff"
    assert p["page"] == "webinar registration" and p["segment"] == "smb"
    assert "MQL" not in p["statistical_result"] or "SQL" in p["statistical_result"]
    assert p["segment_findings"].startswith("No segment differed")
    p4 = prefill(exp(analyzed, "EXP-04"))
    assert "device = desktop" in p4["segment_findings"] and p4["segment"] == "smb"
    assert prefill(exp(analyzed, "EXP-06"))["decision"] == "observe_longer"


def test_build_merges_prefill_draft_and_edits(analyzed):
    e = exp(analyzed)
    d = draft(e)
    learning = build(e, d, {"decision": "stop", "what_happened": "Edited by a person"})
    assert learning.decision == "stop" and learning.what_happened == "Edited by a person"
    assert learning.theme == d.theme and learning.should_not_conclude == d.should_not_conclude
    assert learning.statistical_result == e.statistical_result
    assert learning.next_hypothesis_parts["intervention"] == d.next_hypothesis.intervention
    assert build(e, None).what_happened == ""


def test_should_not_conclude_is_required(analyzed, store):
    e = exp(analyzed)
    learning = build(e, draft(e), {"should_not_conclude": "  "})
    assert "'What We Should NOT Conclude' is required." in check(learning)
    with pytest.raises(ValueError, match="NOT Conclude"):
        save(learning, store)
    assert store.get(Learning, learning.id) is None
    with pytest.raises(ValidationError):  # the LLM draft must provide it too
        LearningDraft(**(draft(e).model_dump() | {"should_not_conclude": "no"}))


def test_draft_cannot_contain_numbers(analyzed):
    with pytest.raises(ValidationError):
        LearningDraft(**(draft(exp(analyzed)).model_dump() | {"what_happened": "Leads rose 35%."}))


def test_send_next_hypothesis_to_validator(analyzed, store):
    e = exp(analyzed)
    learning = build(e, draft(e))
    save(learning, store)
    h = send_next_hypothesis(learning, store)
    assert h.id == "elc-next-EXP-02" and not h.missing_parts and h.title
    assert store.get(Learning, learning.id).next_hypothesis_id == h.id
    assert store.module_of(Hypothesis, h.id) == "experiment_learning_capture"
    from gios.modules.hypothesis_evidence_validator import validate

    assert validate(h, []).total == 0  # no evidence linked yet, manual scoring


def test_demo_cache_covers_every_experiment(analyzed):
    ids = {e.id for e in analyzed.list(Experiment)}
    files = {p.stem.split("__")[1] for p in config.DEMO_DIR.glob("draft_experiment_learning__*.json")}
    assert files == ids


def test_markdown_matches_spec_template(analyzed):
    e = exp(analyzed)
    md = to_markdown(build(e, draft(e)))
    assert [l for l in md.splitlines() if l.startswith("### ")] == [
        "### Experiment", "### Original Problem", "### Hypothesis", "### What Happened",
        "### Statistical / Directional Result", "### Segment Findings", "### What We Learned About the Customer",
        "### What We Learned About the Journey", "### What We Learned About the Business",
        "### What We Should NOT Conclude", "### Decision", "### Reusable Principle", "### Next Hypothesis"]
    assert "**Iterate**" in md


# --- library ---


@pytest.fixture(scope="module")
def learnings(analyzed):
    draft_all(analyzed)
    return analyzed.list(Learning)


def test_library_search_and_facets(learnings):
    assert len(learnings) == 7
    assert [x.experiment_id for x in library.search(learnings, "intent question")] == ["EXP-02"]
    assert [x.experiment_id for x in library.search(learnings, pages=["pricing"])] == ["EXP-03", "EXP-07"]
    assert {x.experiment_id for x in library.search(learnings, themes=["unclear_value"])} == {"EXP-01", "EXP-05"}
    assert [x.experiment_id for x in library.search(learnings, decisions=["observe_longer"])] == ["EXP-06", "EXP-07"]
    assert {x.experiment_id for x in library.search(learnings, segments=["smb"])} >= {"EXP-02"}
    f = library.facets(learnings)
    assert "webinar registration" in f["page"] and "stop" in f["decision"]


def test_library_losses_and_inconclusive_filter(learnings):
    assert {x.experiment_id for x in library.search(learnings, losses_only=True)} == {"EXP-03", "EXP-05"}
    assert library.search(learnings, "pricing", losses_only=True)[0].experiment_id == "EXP-03"


# --- pages ---


def test_capture_page_blocks_save_without_not_conclude_and_sends_hypothesis():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(config.ROOT / "views" / "9_Experiment_Learning_Capture.py")).run(timeout=60)
    at.button(key="elc_seed").click().run(timeout=60)
    at.selectbox(key="elc_pick").select("EXP-02").run(timeout=60)
    assert not at.exception
    at.text_area(key="elc_EXP-02_should_not_conclude").input("").run(timeout=60)
    at.button[0].click().run(timeout=60)  # form submit
    assert any("Saving is blocked" in e.value for e in at.error)
    assert Store().get(Learning, "elc-EXP-02") is None
    at.text_area(key="elc_EXP-02_should_not_conclude").input("That short forms are always bad.").run(timeout=60)
    at.button[0].click().run(timeout=60)
    assert Store().get(Learning, "elc-EXP-02").should_not_conclude == "That short forms are always bad."
    at.button(key="elc_send").click().run(timeout=60)
    assert Store().get(Hypothesis, "elc-next-EXP-02") is not None
    assert at.session_state.hev_pick == "elc-next-EXP-02"


def test_library_page_filters():
    from streamlit.testing.v1 import AppTest

    store = Store()
    analyze_all(store)
    draft_all(store)
    at = AppTest.from_file(str(config.ROOT / "views" / "10_Experiment_Library.py")).run(timeout=60)
    cards = lambda: [e for e in at.expander if not e.label.startswith("How ")]  # noqa: E731
    assert not at.exception and len(cards()) == 7
    at.toggle(key="lib_losses").set_value(True).run(timeout=60)
    assert len(cards()) == 2
    at.text_input(key="lib_query").input("pricing").run(timeout=60)
    assert len(cards()) == 1
