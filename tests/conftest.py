import pytest

from gios.core.store import Store


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "test.db")


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch, tmp_path):
    """Tests always run in DEMO MODE unless a test opts in explicitly, and never touch
    the real gios.db."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("gios.config.DB_PATH", tmp_path / "default.db")


class FakeClient:
    """Stands in for anthropic.Anthropic(); replies are strings or callables(kwargs)->str."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        from types import SimpleNamespace

        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        text = reply(kwargs) if callable(reply) else reply
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=text)])
