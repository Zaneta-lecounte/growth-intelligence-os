import math

import pytest
from pydantic import BaseModel, ValidationError

from gios.core import data, stats
from gios.core.report import Narrative, md_table
from gios.core.schemas import Signal


# --- stats ---


def test_wilson_matches_reference_values():
    lo, hi = stats.wilson_ci(10, 100)
    assert lo == pytest.approx(0.0552, abs=1e-4)
    assert hi == pytest.approx(0.1744, abs=1e-4)


def test_wilson_edges():
    assert stats.wilson_ci(0, 0) == (0.0, 1.0)
    lo, hi = stats.wilson_ci(0, 50)
    assert lo == 0.0 and 0 < hi < 0.1
    lo, hi = stats.wilson_ci(50, 50)
    assert hi == 1.0 and lo > 0.9
    with pytest.raises(ValueError):
        stats.wilson_ci(5, 3)


def test_wilson_narrows_with_n():
    small = stats.wilson_ci(5, 50)
    large = stats.wilson_ci(500, 5000)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_two_proportion_z():
    assert stats.two_proportion_z(50, 100, 50, 100) == 0
    z = stats.two_proportion_z(60, 100, 40, 100)
    assert z == pytest.approx(2.828, abs=1e-3)
    assert stats.two_proportion_z(1, 0, 1, 1) == 0


def test_one_sample_z():
    assert stats.one_sample_z(50, 100, 0.5) == 0
    assert stats.one_sample_z(60, 100, 0.5) == pytest.approx(2.0)
    assert stats.one_sample_z(1, 10, 0) == 0


def test_standardized_rate():
    # 80% mobile at 20%, 20% desktop at 60% -> 28%
    assert stats.standardized_rate([80, 20], [0.2, 0.6]) == pytest.approx(0.28)
    assert stats.standardized_rate([0, 0], [0.2, 0.6]) == 0


def test_frequency_score_bins():
    total = 1000
    assert [stats.frequency_score(c, total) for c in [0, 10, 20, 49, 50, 99, 100, 199, 200, 900]] == \
        [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]


def test_score_to_strength_and_helpers():
    assert stats.score_to_strength(0, [1, 2, 3, 4]) == 1
    assert stats.score_to_strength(3.5, [1, 2, 3, 4]) == 4
    assert stats.pct_deviation(1.5, 1.0) == 0.5
    assert stats.pct_deviation(1, 0) == 0
    assert stats.safe_rate(1, 0) == 0


# --- report ---


class N(BaseModel):
    text: Narrative


def test_narrative_rejects_digits():
    assert N(text="Webinar MQL→SQL is the weakest stage.").text
    with pytest.raises(ValidationError):
        N(text="Webinar converts at 11%.")


def test_md_table_alignment_and_escape():
    out = md_table(["A", "B"], [["x|y", 3]], align=["l", "r"])
    assert out.splitlines()[1] == "|---|---:|"
    assert "x\\|y" in out


# --- data ---


def test_load_adds_ids_and_keys():
    ev = data.load("customer_evidence")
    assert ev.evidence_id.is_unique
    assert ev.text_key.str.match(r"^v[0-9a-f]{10}$").all()
    assert data.text_key(" a ") == data.text_key("a")
    sales = data.load("sales_feedback")
    assert sales.rejection_reason.isna().sum() == 0
    with pytest.raises(KeyError):
        data.load("nope")


# --- store ---


def test_replace_module_output(store):
    def s(summary):
        return Signal(type="funnel", evidence_status="observed", source="x", segment="all",
                      journey_stage="evaluation", summary=summary, strength=3)

    store.save(s("keep"), module="other")
    store.replace_module_output([s("a"), s("b")], module="leakage")
    store.replace_module_output([s("c")], module="leakage")
    assert sorted(x.summary for x in store.list(Signal)) == ["c", "keep"]
    assert store.delete_by_module(Signal, "leakage") == 1
