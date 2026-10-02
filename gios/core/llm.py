"""Single entry point for every LLM call in GIOS.

- Prompts are versioned files: prompts/<name>.v<N>.md with "## System" and "## User"
  sections. The user section is a string.Template ($var placeholders).
- Responses must be JSON only and are validated against a Pydantic model. On a parse or
  validation failure the error is sent back to the model and the call is retried once.
- DEMO MODE (no ANTHROPIC_API_KEY): the cached output in demo/<name>.json (or
  demo/<name>__<demo_key>.json) is validated and returned; no network call is made.

The LLM tags qualitative evidence and writes narrative. It never computes numbers; callers
compute all metrics in Python and pass them in as context.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Mapping, Optional, TypeVar

from pydantic import BaseModel, ValidationError

from gios import config

T = TypeVar("T", bound=BaseModel)

_PROMPT_FILE = re.compile(r"^(?P<name>.+)\.v(?P<version>\d+)\.md$")
_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)

JSON_INSTRUCTIONS = """\
Respond with a single JSON object only: no prose, no markdown fences.
It must validate against this JSON Schema:
{schema}
Do not compute or invent metrics. Use only numbers that appear in the input."""


class LLMError(RuntimeError):
    """Raised when the LLM (or demo cache) cannot produce a valid response."""


@dataclass(frozen=True)
class Prompt:
    name: str
    version: int
    system: str
    user: str

    def render(self, variables: Mapping[str, Any]) -> str:
        try:
            return Template(self.user).substitute(variables)
        except KeyError as exc:
            raise LLMError(f"prompt {self.name} v{self.version} missing variable {exc}") from None


def load_prompt(name: str, version: Optional[int] = None, prompts_dir: Optional[Path] = None) -> Prompt:
    """Load prompts/<name>.v<version>.md; latest version when version is None."""
    prompts_dir = Path(prompts_dir or config.PROMPTS_DIR)
    versions = {
        int(m["version"]): p
        for p in prompts_dir.glob(f"{name}.v*.md")
        if (m := _PROMPT_FILE.match(p.name)) and m["name"] == name
    }
    if not versions:
        raise LLMError(f"no prompt file for '{name}' in {prompts_dir}")
    chosen = max(versions) if version is None else version
    if chosen not in versions:
        raise LLMError(f"prompt '{name}' has no version {chosen}")
    text = versions[chosen].read_text()
    sections = re.split(r"^## (System|User)\s*$", text, flags=re.MULTILINE)
    parts = {sections[i]: sections[i + 1].strip() for i in range(1, len(sections) - 1, 2)}
    if "System" not in parts or "User" not in parts:
        raise LLMError(f"prompt '{name}' v{chosen} needs '## System' and '## User' sections")
    return Prompt(name=name, version=chosen, system=parts["System"], user=parts["User"])


def parse_json_response(text: str, output_model: type[T]) -> T:
    """Strip optional code fences and validate against the model (raises ValidationError)."""
    text = text.strip()
    if m := _FENCE.match(text):
        text = m.group(1)
    return output_model.model_validate_json(text)


def load_demo(prompt_name: str, output_model: type[T], demo_key: Optional[str] = None,
              demo_dir: Optional[Path] = None) -> T:
    demo_dir = Path(demo_dir or config.DEMO_DIR)
    filename = f"{prompt_name}__{demo_key}.json" if demo_key else f"{prompt_name}.json"
    path = demo_dir / filename
    if not path.exists():
        raise LLMError(f"DEMO MODE: no cached output at {path}")
    try:
        return parse_json_response(path.read_text(), output_model)
    except ValidationError as exc:
        raise LLMError(f"DEMO MODE: cached output {path} is invalid: {exc}") from exc


def _response_text(response: Any) -> str:
    if getattr(response, "stop_reason", None) == "refusal":
        raise LLMError("model declined the request (stop_reason=refusal)")
    return "".join(b.text for b in response.content if getattr(b, "type", None) == "text")


def _default_client():
    import anthropic

    return anthropic.Anthropic()


def complete_json(
    prompt_name: str,
    output_model: type[T],
    variables: Optional[Mapping[str, Any]] = None,
    *,
    version: Optional[int] = None,
    demo_key: Optional[str] = None,
    client: Any = None,
    prompts_dir: Optional[Path] = None,
    demo_dir: Optional[Path] = None,
) -> T:
    """Run a versioned prompt and return a validated `output_model` instance."""
    if config.is_demo_mode():
        return load_demo(prompt_name, output_model, demo_key, demo_dir)

    prompt = load_prompt(prompt_name, version, prompts_dir)
    schema = json.dumps(output_model.model_json_schema(), indent=2)
    system = f"{prompt.system}\n\n{JSON_INSTRUCTIONS.format(schema=schema)}"
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt.render(variables or {})}]
    client = client or _default_client()

    last_error: Optional[Exception] = None
    for attempt in range(2):  # first try + one retry on validation failure
        response = client.messages.create(
            model=config.MODEL,
            max_tokens=config.MAX_TOKENS,
            system=system,
            messages=messages,
        )
        text = _response_text(response)
        try:
            return parse_json_response(text, output_model)
        except ValidationError as exc:
            last_error = exc
            messages = messages + [
                {"role": "assistant", "content": text or "(empty)"},
                {
                    "role": "user",
                    "content": "Your response failed validation:\n"
                    f"{exc}\nReturn the corrected JSON object only.",
                },
            ]
    raise LLMError(f"{prompt_name} v{prompt.version}: invalid JSON after retry: {last_error}")
