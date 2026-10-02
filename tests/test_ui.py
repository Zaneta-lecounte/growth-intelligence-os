"""Workspace isolation and graceful degradation in the Streamlit layer."""
import json

from streamlit.testing.v1 import AppTest

from gios import config
from gios.core.schemas import Signal
from gios.ui import recommendation_badge, evidence_badge

DIAG = str(config.ROOT / "views" / "4_Growth_Intelligence_Diagnostic.py")


def test_session_store_isolates_visitors(monkeypatch):
    monkeypatch.setenv("GIOS_STORE_MODE", "session")
    alice = AppTest.from_file(DIAG).run(timeout=60)
    alice.button(key="gid_seed").click().run(timeout=120)
    assert not alice.exception
    assert len(alice.session_state.gios_db_path) > 0
    from gios.core.store import Store

    assert len(Store(alice.session_state.gios_db_path).list(Signal)) == 26
    bob = AppTest.from_file(DIAG).run(timeout=60)
    assert bob.session_state.gios_db_path != alice.session_state.gios_db_path
    assert bob.button(key="gid_seed")  # Bob starts with an empty workspace
    assert len(Store().list(Signal)) == 0  # the shared store was never touched


def test_pages_degrade_when_the_api_fails(monkeypatch):
    """With a key set but the API unreachable, measured results still render."""
    import anthropic
    import httpx
    import streamlit as st

    st.cache_data.clear()  # earlier tests cached demo results for these pages

    def down(self, **kwargs):
        raise anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(anthropic.resources.messages.Messages, "create", down)
    leak = AppTest.from_file(str(config.ROOT / "views" / "3_Qualified_Demand_Leakage_Auditor.py")).run(timeout=60)
    assert not leak.exception
    assert any("Narrative unavailable" in m.value for m in leak.markdown)
    assert any("webinar · MQL→SQL" in m.value for m in leak.markdown)
    friction = AppTest.from_file(str(config.ROOT / "views" / "2_Behavioral_Friction_Analyzer.py")).run(timeout=60)
    assert not friction.exception
    assert any("Friction classification unavailable" in m.value for m in friction.markdown)
    css = AppTest.from_file(str(config.ROOT / "views" / "1_Customer_Signal_Synthesizer.py")).run(timeout=60)
    assert not css.exception
    assert any("Could not tag the customer evidence" in e.value for e in css.error)


def test_badges():
    assert evidence_badge("observed") == ":green-badge[Observed]"
    assert recommendation_badge("run_now", "Run now") == ":green-badge[Run now]"
    assert recommendation_badge("stop") == ":red-badge[Stop]"


def test_every_module_page_has_how_it_works():
    for view in sorted((config.ROOT / "views").glob("*.py")):
        at = AppTest.from_file(str(view)).run(timeout=60)
        assert not at.exception, view.name
        assert any(e.label.startswith("How ") and "works" in e.label for e in at.expander), view.name
