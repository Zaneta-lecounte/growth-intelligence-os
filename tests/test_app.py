from streamlit.testing.v1 import AppTest

from gios import config
from gios.modules import LAYERS, MODULES


def test_registry_matches_specs():
    specs = {p.name for p in config.SPECS_DIR.glob("*.md")}
    assert len(MODULES) == 10
    assert {m.spec for m in MODULES} == specs
    assert len({m.slug for m in MODULES}) == 10
    assert {m.layer for m in MODULES} == set(LAYERS)


def test_every_module_has_a_package():
    import importlib

    for m in MODULES:
        importlib.import_module(f"gios.modules.{m.slug}")


def test_home_page_renders_in_demo_mode():
    at = AppTest.from_file(str(config.ROOT / "app.py")).run(timeout=30)
    assert not at.exception
    assert any("Demo mode" in i.value for i in at.info)
    assert "GIOS" in at.title[0].value


def test_home_page_hides_banner_with_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    at = AppTest.from_file(str(config.ROOT / "app.py")).run(timeout=30)
    assert not at.exception
    assert not any("Demo mode" in i.value for i in at.info)
