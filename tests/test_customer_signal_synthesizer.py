import json
import re

import pandas as pd
import pytest

from gios import config
from gios.core import data
from gios.core.schemas import Signal
from gios.core.store import Store
from gios.modules.customer_signal_synthesizer import analysis, pipeline, run
from gios.modules.customer_signal_synthesizer.models import EvidenceTag, TagBatch, ThemeAssessment, ThemeSynthesis
from gios.modules.customer_signal_synthesizer.report import to_markdown
from tests.conftest import FakeClient


def tag(key, theme="pricing_uncertainty", segment="unknown", stage="unknown", relevance="high"):
    return EvidenceTag(key=key, theme=theme, topic="t", sentiment="negative", segment=segment,
                       journey_stage=stage, business_relevance=relevance)


def evidence_frame(rows):
    df = pd.DataFrame(rows, columns=["date", "source", "segment", "journey_stage", "verbatim"])
    df.insert(0, "evidence_id", [f"E{i}" for i in range(len(df))])
    df["text_key"] = df.verbatim.map(data.text_key)
    df["month"] = df.date.str[:7]
    return df


@pytest.fixture(scope="module")
def demo_result():
    return run()


# --- deterministic helpers ---


def test_unique_verbatims_and_batches():
    ev = evidence_frame([
        ["2026-03-01", "chat", "smb", "evaluation", "A"],
        ["2026-03-02", "survey", "mid_market", "purchase", "A"],
        ["2026-03-03", "chat", "smb", "evaluation", "B"],
    ])
    u = analysis.unique_verbatims(ev)
    assert len(u) == 2
    a = u.set_index("verbatim").loc["A"]
    assert a.mentions == 2 and a.sources == ["chat", "survey"] and a.segments == ["mid_market", "smb"]
    assert [len(b) for b in analysis.batches(pd.concat([u] * 50), size=40)] == [40, 40, 20]
    lines = analysis.batch_payload(u).splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["key"] == u.key.iloc[0]


def test_merge_tags_filters_and_dedupes():
    merged = analysis.merge_tags([tag("a"), tag("a", theme="urgency"), tag("x")], wanted_keys={"a"})
    assert list(merged) == ["a"] and merged["a"].theme == "pricing_uncertainty"


def test_apply_tags_metadata_wins_and_disagreements():
    ev = evidence_frame([
        ["2026-03-01", "chat", "smb", "evaluation", "A"],
        ["2026-03-01", "chat", "smb", "consideration", "B"],
        ["2026-03-01", "chat", "smb", "retention", "C"],
        ["2026-03-01", "chat", "smb", "onboarding", "D"],
    ])
    keys = dict(zip(ev.verbatim, ev.text_key))
    tags = {keys["A"]: tag(keys["A"], segment="mid_market", stage="purchase"),   # seg differs, same phase
            keys["B"]: tag(keys["B"], stage="evaluation"),                       # neighbour stage: fine
            keys["C"]: tag(keys["C"], stage="evaluation")}                       # post vs pre: counts
    out = analysis.apply_tags(ev, tags)
    assert out.segment.tolist() == ["smb"] * 4
    assert out.tagged.tolist() == [True, True, True, False]
    assert analysis.disagreements(out) == {"segment": 1, "journey_stage": 1}


def _tagged_frame(theme_counts_by_month):
    rows, tags = [], {}
    for theme, monthly in theme_counts_by_month.items():
        for m, n in enumerate(monthly):
            for i in range(n):
                text = f"{theme} {m} {i}"
                rows.append([f"2026-0{m + 3}-01", "chat", "smb", "evaluation", text])
    ev = evidence_frame(rows)
    for r in ev.itertuples():
        tags[r.text_key] = tag(r.text_key, theme=r.verbatim.split()[0])
    return analysis.apply_tags(ev, tags)


def test_theme_table_counts_bins_and_trend():
    tagged = _tagged_frame({"pricing_uncertainty": [2, 2, 4, 6, 8, 10], "usability_polish": [10] * 6,
                            "urgency": [1, 0, 0, 0, 0, 0]})
    t = analysis.theme_table(tagged).set_index("theme")
    assert t.loc["usability_polish", "count"] == 60
    assert t.loc["pricing_uncertainty", "count"] == 32
    assert t.loc["usability_polish", "frequency"] == 5            # 60/93 = 65%
    assert t.loc["urgency", "frequency"] == 1                     # 1/93 ≈ 1%
    assert t.loc["pricing_uncertainty", "trend"] == "rising"
    assert t.loc["usability_polish", "trend"] == "stable"
    assert t.loc["urgency", "trend"] == "stable"                  # too few to call
    assert t.loc["pricing_uncertainty", "segment"] == "smb"


def test_representative_quote_prefers_relevance_then_frequency():
    g = pd.DataFrame({"verbatim": ["low", "low", "low", "high b", "high a", "high a"],
                      "business_relevance": ["low"] * 3 + ["high"] * 3})
    assert analysis.representative_quote(g) == "high a"


def test_rank_formula_and_order():
    themes = pd.DataFrame({"theme": ["a", "b"], "count": [100, 10], "frequency": [5, 2],
                           "severity": [1, 5], "commercial_relevance": [1, 5], "journey_relevance": [2, 4]})
    r = analysis.rank(themes).set_index("theme")
    assert r.loc["a", "overall"] == 5 * 1 * 1 * 2
    assert r.loc["b", "overall"] == 2 * 5 * 5 * 4
    assert r.loc["b", "overall_rank"] == 1 and r.loc["a", "frequency_rank"] == 1
    assert r.loc["b", "severity_x_commercial"] == 25


def test_attach_scores_defaults_missing_theme():
    themes = pd.DataFrame({"theme": ["a", "b"], "count": [1, 1], "frequency": [1, 1]})
    a = ThemeAssessment(theme="pricing_uncertainty", severity=5, commercial_relevance=5, journey_relevance=5,
                        summary="s", expected_behavior="e", funnel_stage="f", analytics_signal="a",
                        testable_question="q")
    themes.loc[0, "theme"] = "pricing_uncertainty"
    out = analysis.attach_scores(themes, [a]).set_index("theme")
    assert out.loc["b", "severity"] == analysis.DEFAULT_SCORE and not out.loc["b", "assessed"]
    assert out.loc["pricing_uncertainty", "overall"] == 125


# --- pipeline with a fake LLM (live mode path) ---


def test_tagging_is_batched_and_missing_keys_untagged(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    ev = evidence_frame([["2026-03-01", "chat", "smb", "evaluation", f"verbatim {i}"] for i in range(45)])

    def reply(kwargs):
        keys = re.findall(r'"key": "(v[0-9a-f]+)"', kwargs["messages"][0]["content"])
        return TagBatch(tags=[tag(k) for k in keys[:-1]]).model_dump_json()  # drop one key per batch

    client = FakeClient([reply, reply])
    tags = pipeline.tag_evidence(ev, client=client)
    assert len(client.calls) == 2
    assert len(tags) == 43
    assert (~analysis.apply_tags(ev, tags).tagged).sum() == 2


# --- demo cache and acceptance criteria ---


def test_demo_cache_covers_every_distinct_verbatim():
    cached = TagBatch.model_validate_json((config.DEMO_DIR / "tag_customer_evidence.json").read_text())
    ev = data.load("customer_evidence")
    assert {t.key for t in cached.tags} == set(ev.text_key)
    synth = ThemeSynthesis.model_validate_json((config.DEMO_DIR / "synthesize_customer_themes.json").read_text())
    assert synth.tensions and synth.research_gaps


def test_demo_run_tags_everything(demo_result):
    assert demo_result.untagged == 0
    assert demo_result.themes.assessed.all()
    assert demo_result.themes["count"].sum() == demo_result.meta["evidence_rows"]


def test_acceptance_pricing_ranks_high_on_severity_x_commercial(demo_result):
    t = demo_result.themes.set_index("theme")
    assert t.loc["pricing_uncertainty", "overall_rank"] == 1
    assert t.loc["pricing_uncertainty", "severity_x_commercial"] == t.severity_x_commercial.max()
    assert t.loc["pricing_uncertainty", "trend"] == "rising"


def test_acceptance_red_herring_frequent_but_not_overall(demo_result):
    t = demo_result.themes.set_index("theme")
    assert t.loc["usability_polish", "frequency_rank"] == 1
    assert t.loc["usability_polish", "frequency"] == 5
    assert t.loc["usability_polish", "overall_rank"] > len(t) / 2


def test_signals(demo_result, store):
    signals = analysis.to_signals(demo_result.themes)
    assert len(signals) == len(demo_result.themes)
    by_id = {s.id: s for s in signals}
    assert by_id["css-pricing_uncertainty"].strength == 5
    assert by_id["css-usability_polish"].strength == 1
    assert all(s.type == "customer" for s in signals)
    store.replace_module_output(signals, "customer_signal_synthesizer")
    assert len(store.list(Signal, module="customer_signal_synthesizer")) == len(signals)


def test_markdown_matches_spec_format(demo_result):
    md = to_markdown(demo_result.themes, demo_result.synthesis, demo_result.untagged,
                     demo_result.disagreements, demo_result.meta)
    headings = [l for l in md.splitlines() if l.startswith("### ")]
    assert headings == ["### Top Customer Themes", "### Customer Tensions", "### Behavioral Signals to Validate",
                        "### Growth Opportunities", "### Research Gaps"]
    assert "| Theme | Evidence | Segment | Journey Stage | Frequency | Severity |" in md
    assert "|---|---|---|---|---:|---:|" in md
    assert md.index("Pricing uncertainty") < md.index("Usability polish")


def test_page_renders_and_saves(monkeypatch):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(config.ROOT / "views" / "1_Customer_Signal_Synthesizer.py")).run(timeout=60)
    assert not at.exception
    assert any("Top Customer Themes" in m.value for m in at.markdown)
    at.button(key="css_save").click().run(timeout=60)
    assert not at.exception
    assert len(Store().list(Signal, module="customer_signal_synthesizer")) == 10
