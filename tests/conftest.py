import pytest

from gios.core.store import Store


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "test.db")


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch):
    """Tests always run in DEMO MODE unless a test opts in explicitly."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
