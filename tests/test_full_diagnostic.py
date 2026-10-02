"""Integration: the one-click run surfaces all three embedded stories as Now priorities and keeps
the frequent-but-low-severity red herring off the roadmap."""
import pytest

from gios import config
from gios.core.schemas import Experiment, Opportunity, Signal
from gios.core.store import Store
from gios.modules import full_diagnostic
from gios.modules.growth_priority_orchestrator.report import to_markdown


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    store = Store(tmp_path_factory.mktemp("full") / "s.db")
    steps = []
    run = full_diagnostic.run(store, on_step=lambda i, key, label, n: steps.append((key, n)))
    return store, run, steps


def test_runs_every_layer_in_order(result):
    _, run, steps = result
    assert [k for k, _ in steps] == ["signals", "diagnostics", "validations", "backlog", "roadmap"]
    assert run.counts == {"signals": 26, "diagnostics": 7, "validations": 7, "backlog": 7, "roadmap": 5}


@pytest.mark.parametrize("story", ["pricing", "webinar", "mobile"])
def test_each_story_is_a_now_priority_with_its_full_evidence(result, story):
    _, run, _ = result
    s = next(x for x in run.stories if x.key == story)
    assert s.complete, (story, s.matched)
    assert s.prioritized, (story, s.item.horizon)


def test_pricing_story_links_verbatims_exits_and_cac(result):
    _, run, _ = result
    pricing = next(x for x in run.stories if x.key == "pricing")
    assert pricing.matched == ["pricing-clarity verbatims", "pricing-page exits for paid search",
                               "rising paid search CAC"]


def test_red_herring_is_most_frequent_but_not_prioritized(result):
    store, run, _ = result
    h = run.red_herring
    assert h.signal is not None and h.frequency_rank == 1
    assert h.overall_rank > 5 and h.deprioritized
    now = [o for o in run.roadmap.active if o.horizon == "now"]
    assert all(h.signal.id not in o.signal_ids for o in run.roadmap.active)
    assert len(now) == 3 and run.passed


def test_executive_priorities_name_the_three_stories(result):
    _, run, _ = result
    md = to_markdown(run.roadmap)
    section = md.split("### Executive Growth Priorities")[1].split("### Now")[0]
    for title in ["Clarify pricing for paid search evaluators", "Fix the mobile demo form",
                  "Webinar leads are learners, not buyers"]:
        assert title in section
    assert "dark mode" not in md.lower() and "usability polish" not in md.lower()


def test_rerun_is_idempotent_and_keeps_experiments(result):
    store, first, _ = result
    store.save(Experiment(id="keep-me", name="x", variants=[
        {"name": "a", "visitors": 1, "conversions": 0}, {"name": "b", "visitors": 1, "conversions": 0}]))
    again = full_diagnostic.run(store)
    assert again.counts == first.counts and again.passed
    assert store.get(Experiment, "keep-me") is not None
    assert full_diagnostic.latest(store).passed


def test_page_runs_and_reports_story_checks():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(config.ROOT / "views" / "0_Run_Full_Diagnostic.py")).run(timeout=60)
    assert not at.exception
    at.button(key="full_run").click().run(timeout=240)
    assert not at.exception
    assert any("All three embedded stories" in s.value for s in at.success)
    assert any("Executive Growth Priorities" in m.value for m in at.markdown)
    assert len(Store().list(Signal)) == 26 and any(o.horizon == "now" for o in Store().list(Opportunity))
