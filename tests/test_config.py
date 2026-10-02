from gios import config


def test_demo_mode_follows_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert config.is_demo_mode()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "  ")
    assert config.is_demo_mode()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert not config.is_demo_mode()


def test_model_is_sonnet():
    assert "sonnet" in config.MODEL
