import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, Field

from gios.core import llm
from gios.core.llm import LLMError, complete_json, load_prompt


class Tag(BaseModel):
    theme: str
    severity: int = Field(ge=1, le=5)


PROMPT = """## System
You tag customer evidence.

## User
Tag this verbatim: $verbatim
"""


@pytest.fixture
def dirs(tmp_path):
    prompts, demo = tmp_path / "prompts", tmp_path / "demo"
    prompts.mkdir()
    demo.mkdir()
    (prompts / "tag.v1.md").write_text(PROMPT)
    (prompts / "tag.v2.md").write_text(PROMPT.replace("You tag", "V2: you tag"))
    return SimpleNamespace(prompts=prompts, demo=demo)


class FakeClient:
    def __init__(self, replies, stop_reason="end_turn"):
        self.replies = list(replies)
        self.calls = []
        self.stop_reason = stop_reason
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        text = self.replies.pop(0)
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        )


def live(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")


# --- prompts ---


def test_load_prompt_latest_and_pinned(dirs):
    assert load_prompt("tag", prompts_dir=dirs.prompts).version == 2
    p = load_prompt("tag", version=1, prompts_dir=dirs.prompts)
    assert p.system == "You tag customer evidence."
    assert p.render({"verbatim": "too pricey"}) == "Tag this verbatim: too pricey"


def test_load_prompt_errors(dirs):
    with pytest.raises(LLMError):
        load_prompt("missing", prompts_dir=dirs.prompts)
    with pytest.raises(LLMError):
        load_prompt("tag", version=9, prompts_dir=dirs.prompts)
    (dirs.prompts / "bad.v1.md").write_text("no sections")
    with pytest.raises(LLMError):
        load_prompt("bad", prompts_dir=dirs.prompts)
    with pytest.raises(LLMError):
        load_prompt("tag", prompts_dir=dirs.prompts).render({})


# --- demo mode ---


def test_demo_mode_returns_cached_output_without_client(dirs):
    (dirs.demo / "tag.json").write_text(json.dumps({"theme": "pricing", "severity": 4}))

    class Boom:
        @property
        def messages(self):
            raise AssertionError("no API call in demo mode")

    out = complete_json("tag", Tag, {"verbatim": "x"}, client=Boom(),
                        prompts_dir=dirs.prompts, demo_dir=dirs.demo)
    assert out == Tag(theme="pricing", severity=4)


def test_demo_mode_keyed_cache(dirs):
    (dirs.demo / "tag__mobile.json").write_text('{"theme": "technical", "severity": 5}')
    out = complete_json("tag", Tag, demo_key="mobile", demo_dir=dirs.demo)
    assert out.theme == "technical"


def test_demo_mode_missing_or_invalid_cache(dirs):
    with pytest.raises(LLMError, match="no cached output"):
        complete_json("tag", Tag, demo_dir=dirs.demo)
    (dirs.demo / "tag.json").write_text('{"theme": "pricing", "severity": 9}')
    with pytest.raises(LLMError, match="invalid"):
        complete_json("tag", Tag, demo_dir=dirs.demo)


# --- live mode (fake client) ---


def test_live_valid_json_first_try(dirs, monkeypatch):
    live(monkeypatch)
    client = FakeClient(['{"theme": "pricing", "severity": 3}'])
    out = complete_json("tag", Tag, {"verbatim": "x"}, client=client, prompts_dir=dirs.prompts)
    assert out.severity == 3
    call = client.calls[0]
    assert call["model"] == llm.config.MODEL
    assert "V2: you tag" in call["system"] and '"severity"' in call["system"]
    assert call["messages"] == [{"role": "user", "content": "Tag this verbatim: x"}]


def test_live_strips_code_fences(dirs, monkeypatch):
    live(monkeypatch)
    client = FakeClient(['```json\n{"theme": "trust", "severity": 2}\n```'])
    assert complete_json("tag", Tag, {"verbatim": "x"}, client=client,
                         prompts_dir=dirs.prompts).theme == "trust"


def test_live_retries_once_with_validation_error(dirs, monkeypatch):
    live(monkeypatch)
    client = FakeClient(["not json", '{"theme": "pricing", "severity": 4}'])
    out = complete_json("tag", Tag, {"verbatim": "x"}, client=client, prompts_dir=dirs.prompts)
    assert out.severity == 4
    assert len(client.calls) == 2
    retry_msgs = client.calls[1]["messages"]
    assert retry_msgs[1] == {"role": "assistant", "content": "not json"}
    assert "failed validation" in retry_msgs[2]["content"]


def test_live_fails_after_one_retry(dirs, monkeypatch):
    live(monkeypatch)
    client = FakeClient(['{"theme": "x", "severity": 0}', '{"theme": "x"}', "unused"])
    with pytest.raises(LLMError, match="after retry"):
        complete_json("tag", Tag, {"verbatim": "x"}, client=client, prompts_dir=dirs.prompts)
    assert len(client.calls) == 2


def test_live_refusal_raises(dirs, monkeypatch):
    live(monkeypatch)
    client = FakeClient(["{}"], stop_reason="refusal")
    with pytest.raises(LLMError, match="refusal"):
        complete_json("tag", Tag, {"verbatim": "x"}, client=client, prompts_dir=dirs.prompts)


# --- shipped prompts and demo cache stay consistent ---


def test_every_shipped_prompt_parses():
    for path in llm.config.PROMPTS_DIR.glob("*.v*.md"):
        name = path.name.split(".v")[0]
        load_prompt(name)


def test_every_demo_file_is_json():
    for path in llm.config.DEMO_DIR.glob("*.json"):
        json.loads(path.read_text())
