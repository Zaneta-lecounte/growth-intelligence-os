# Prompts

One file per LLM step, versioned: `<step>.v<N>.md`. Never edit a released version; add `v<N+1>`.
`gios.core.llm.load_prompt` picks the latest version unless a version is pinned.

Each file has two sections:

```
## System
Role and rules. The JSON-only instruction and output schema are appended by llm.py.

## User
Template using $placeholders (string.Template). Pass precomputed numbers in as context.
The LLM never computes numbers.
```

Every prompt needs a cached output in `demo/<step>.json` (or `demo/<step>__<key>.json`) so the
app works in DEMO MODE.
