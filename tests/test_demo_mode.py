"""The public app must work end to end with no API key: every page renders without errors on a
fully seeded workspace, and every default path is served from demo/."""
import pytest
from streamlit.testing.v1 import AppTest

from gios import config
from gios.core.store import Store
from gios.modules.demo_flow import CACHED_DIAGNOSTIC_SCOPES, diagnostic_inputs, seed_through

VIEWS = sorted((config.ROOT / "views").glob("*.py"))


@pytest.fixture
def seeded():
    seed_through("learnings", Store())


@pytest.mark.parametrize("view", VIEWS, ids=lambda p: p.stem)
def test_every_page_renders_without_api_key(seeded, view):
    assert config.is_demo_mode()
    at = AppTest.from_file(str(view)).run(timeout=120)
    assert not at.exception, [e.value for e in at.exception]
    expected = ("Diagnosis blocked",)  # the Step 1 gate is meant to show until the business signal is set
    assert not [e.value for e in at.error if not e.value.startswith(expected)], view.name


def test_home_and_router_render():
    for entry in ("app.py", "home.py"):
        at = AppTest.from_file(str(config.ROOT / entry)).run(timeout=60)
        assert not at.exception


@pytest.mark.parametrize("scope", list(CACHED_DIAGNOSTIC_SCOPES))
def test_every_cached_diagnosis_scope_validates(scope):
    from gios.core.schemas import Signal
    from gios.modules.growth_intelligence_diagnostic import run

    store = Store()
    seed_through("signals", store)
    business, filters = diagnostic_inputs(scope)
    result = run(business, filters, store.list(Signal))
    assert result.ranked and result.leakage_point is not None


def test_uncached_scope_explains_itself(seeded):
    at = AppTest.from_file(str(config.ROOT / "views" / "4_Growth_Intelligence_Diagnostic.py")).run(timeout=60)
    at.selectbox(key="gid_channel").select("organic").run(timeout=60)
    at.button(key="gid_fill").click().run(timeout=60)
    at.button(key="gid_run").click().run(timeout=60)
    assert not at.exception
    assert any("Demo mode has cached diagnoses" in i.value for i in at.info)
