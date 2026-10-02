# CLAUDE.md — GIOS working rules

GIOS (Growth Intelligence Operating System) is a portfolio project: a Streamlit app that
turns fragmented growth signals for a fictional B2B SaaS (EchoAI) into diagnoses,
priorities, and learnings.

## Rules
1. **`specs/` is the source of truth.** Each module's inputs, method, and output format come
   from its spec. Read the spec before touching a module. If code and spec disagree, the spec wins.
2. **Synthetic data only.** All data comes from `scripts/generate_data.py` (seeded, reproducible)
   into `data/synthetic/`. Never add real customer, company, or personal data.
3. **Deterministic math.** All math, scoring, thresholds, and statistics are plain Python with
   tests. The LLM only tags qualitative evidence and writes narrative. The LLM never computes a number.
4. **One LLM entry point.** Every LLM call goes through `gios/core/llm.py`, uses a versioned prompt
   in `prompts/`, and returns JSON validated against a Pydantic model. Each prompt needs a cached
   output in `demo/` so the app runs without an API key (DEMO MODE).
5. **Run tests before every commit:** `python -m pytest -q`. Do not commit red.
6. **LLM text never carries numbers.** Narrative fields use `gios.core.report.Narrative`, which rejects
   digits; reports render computed values next to the prose.
7. **Conventional commits:** `feat:`, `fix:`, `chore:`, `docs:`, `test:`, `refactor:` (optional scope, e.g. `feat(core): ...`).

## Layout
- `app.py` — `st.navigation` router; `home.py` — home page; `views/` — one Streamlit page per module (registered in `app.py`)
- `gios/config.py` — model name, paths, demo-mode detection
- `gios/core/` — `schemas.py` (shared Pydantic models), `llm.py` (LLM wrapper), `store.py` (SQLite)
- `gios/modules/` — one package per module (deterministic logic + LLM narrative)
- `prompts/` — versioned prompt files (`<step>.v<N>.md`); `demo/` — cached LLM outputs
- `data/synthetic/` — generated CSVs; `tests/` — pytest

## Commands
- `pip install -r requirements.txt`
- `python scripts/generate_data.py` — regenerate synthetic data
- `python scripts/build_demo_cache.py [--live] [--only <module>]` — rebuild `demo/` (rerun after changing data or prompts)
- `streamlit run app.py`
- `python -m pytest -q`
