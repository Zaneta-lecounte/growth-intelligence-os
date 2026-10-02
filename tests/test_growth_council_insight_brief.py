import re
import json

import pytest
from pydantic import ValidationError

from gios import config
from gios.core.llm import LLMError
from gios.core.schemas import Opportunity, Signal
from gios.core.store import Store
from gios.modules.demo_flow import seed_through
from gios.modules.growth_council_insight_brief import analysis, gather, run, slack, write
from gios.modules.growth_council_insight_brief.models import checked_brief_model
from gios.modules.growth_council_insight_brief.pipeline import Brief
from gios.modules.growth_council_insight_brief.report import to_markdown
from tests.conftest import FakeClient

CACHE = config.DEMO_DIR / "write_growth_council_brief__full.json"


@pytest.fixture(scope="module")
def seeded(tmp_path_factory):
    store = Store(tmp_path_factory.mktemp("gcb") / "s.db")
    seed_through("learnings", store)
    return store


@pytest.fixture(scope="module")
def brief(seeded):
    return run(seeded, "2026-03", "2026-08")


def test_gather_filters_by_period(seeded):
    full = gather(seeded, "2026-03", "2026-08")
    assert full.counts == {"signals": 26, "hypotheses": 7, "roadmap": 5, "experiments": 7, "learnings": 7}
    summer = gather(seeded, "2026-07", "2026-08")
    assert {e.id for e in summer.experiments} == {"EXP-05", "EXP-06", "EXP-07"}
    assert {x.experiment_id for x in summer.learnings} == {"EXP-05", "EXP-06", "EXP-07"}
    assert gather(seeded, "2027-01", "2027-02").counts["signals"] == 0
    assert [o.horizon for o in full.roadmap] == ["now", "now", "now", "next", "next"]


def test_brief_rejects_unknown_ids(seeded):
    inputs = gather(seeded, "2026-03", "2026-08")
    raw = json.loads(CACHE.read_text())
    model = checked_brief_model(inputs.allowed_ids())
    assert model(**raw)
    raw["signals"]["funnel"]["ids"] = ["qdl-invented"]
    with pytest.raises(ValidationError, match="qdl-invented"):
        model(**raw)


def test_live_brief_retries_then_fails_on_unknown_ids(seeded, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    raw = json.loads(CACHE.read_text())
    raw["learning_ids"] = ["elc-EXP-99"]
    client = FakeClient([json.dumps(raw)] * 2)
    with pytest.raises(LLMError, match="elc-EXP-99"):
        run(seeded, "2026-03", "2026-08", client=client)
    assert len(client.calls) == 2


def test_other_periods_have_no_demo_cache(seeded):
    with pytest.raises(LLMError, match="no cached output"):
        run(seeded, "2026-05", "2026-08")
    with pytest.raises(ValueError, match="No signals"):
        write(gather(seeded, "2027-01", "2027-02"))


def test_owner_suggestions_come_from_leakage_owner_classes(brief):
    assert brief.owner == "web_conversion"
    assert brief.suggestions[:2] == ["web_conversion", "acquisition"]
    inputs = brief.inputs
    assert analysis.owner_suggestions(["qdl-flag-qualification_mismatch-webinar"], inputs) == ["qualification"]
    assert analysis.owner_suggestions(["css-pricing_uncertainty"], inputs) == []


def test_slack_summary_is_at_most_eight_lines(brief):
    text = slack(brief)
    lines = text.splitlines()
    assert len(lines) <= 8
    assert lines[0].startswith(":bar_chart: *Growth Council brief*")
    assert "(owner: Web / conversion)" in text
    brief_named = Brief(brief.inputs, brief.draft, "qualification", owner_name="RevOps team")
    assert "(owner: Qualification (RevOps team))" in slack(brief_named)


def test_first_sentence():
    assert analysis.first_sentence("One. Two.") == "One."
    assert analysis.first_sentence("No full stop") == "No full stop"
    assert len(analysis.first_sentence("x" * 500, limit=50)) == 50


def test_markdown_has_the_ten_spec_sections(brief):
    md = to_markdown(brief)
    assert [l for l in md.splitlines() if l.startswith("### ")] == [
        "### 1. What changed?", "### 2. Why does it matter?", "### 3. What signals support it?",
        "### 4. Working hypothesis", "### 5. Recommended action", "### 6. Cross-functional dependency",
        "### 7. Owner", "### 8. Measurement", "### 9. What should the organization learn?",
        "### 10. Decision needed from Growth Council"]
    for label in ["- Customer:", "- Behavioral:", "- Funnel:", "- Revenue:", "- Sales / operational:"]:
        assert label in md
    cac = brief.inputs.by_id()["qdl-flag-rising_cac-paid_search"].summary
    assert re.search(r"CAC \$[\d,]+ in 2026-06", cac) and cac[:60] in md  # numbers come from the cited signal
    assert "Validator score 11/12" in md and "Roadmap: **Clarify pricing for paid search evaluators**, Now" in md
    assert "Size the test before running it" in md


def test_page_renders_and_owner_is_editable():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(config.ROOT / "views" / "11_Growth_Council_Insight_Brief.py")).run(timeout=60)
    at.button(key="gcb_seed").click().run(timeout=240)
    assert not at.exception
    assert any("### 10. Decision needed" in m.value for m in at.markdown)
    at.selectbox(key="gcb_brief_2026-03_2026-08_owner").select("qualification").run(timeout=60)
    assert "owner: Qualification" in at.code[0].value
    at.select_slider(key="gcb_period").set_value(("2026-05", "2026-08")).run(timeout=60)
    assert any("No brief for this period" in e.value for e in at.error)
