"""Shared Streamlit UI helpers for module pages: store per session, page header with the
"How this module works" expander, badges, report/download, and seeding."""
from __future__ import annotations

import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import streamlit as st

from gios import config
from gios.core.schemas import Signal
from gios.core.store import Store
from gios.modules import ModuleInfo

# Reference palette (dataviz skill): categorical slot 1 and slot 2, plus a recessive neutral.
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"
NEUTRAL = "#a3a29c"

SESSION_DIR = Path(tempfile.gettempdir()) / "gios-sessions"
SESSION_TTL_SECONDS = 24 * 3600


# --- Store ----------------------------------------------------------------------------------------


def store_mode() -> str:
    """'session' (default): every browser session gets its own SQLite file, so visitors to a public
    deployment never see or overwrite each other's work. 'shared': one persistent gios.db."""
    return os.getenv("GIOS_STORE_MODE", "session").strip().lower()


def _cleanup_old_sessions() -> None:
    cutoff = time.time() - SESSION_TTL_SECONDS
    for f in SESSION_DIR.glob("*.db"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
        except OSError:
            pass


def get_store() -> Store:
    if store_mode() == "shared":
        return Store()
    if "gios_db_path" not in st.session_state:
        SESSION_DIR.mkdir(parents=True, exist_ok=True)
        _cleanup_old_sessions()
        st.session_state.gios_db_path = str(SESSION_DIR / f"{uuid.uuid4().hex}.db")
    return Store(st.session_state.gios_db_path)


# --- Badges ---------------------------------------------------------------------------------------

from gios.core.report import EVIDENCE_BADGES_UI as EVIDENCE_BADGES  # noqa: E402
RECOMMENDATION_COLORS = {
    # scorer
    "run_now": "green", "research_first": "orange", "instrument_first": "violet", "defer": "gray", "reject": "red",
    # downstream impact
    "scale": "green", "iterate": "blue", "retest": "orange", "stop": "red", "observe_longer": "gray",
    # validator
    "test": "green",
}


def evidence_badge(status: str) -> str:
    return EVIDENCE_BADGES.get(status, f":gray-badge[{status}]")


def recommendation_badge(key: str, label: Optional[str] = None) -> str:
    color = RECOMMENDATION_COLORS.get(key, "gray")
    return f":{color}-badge[{label or key.replace('_', ' ').capitalize()}]"


# --- Page scaffolding -----------------------------------------------------------------------------


def spec_url(spec: str) -> str:
    return f"{config.REPO_URL}/blob/main/specs/{spec}"


def how_it_works(title: str, deterministic: str, llm: str, specs: list[str]) -> None:
    with st.expander(f"How {title} works"):
        st.markdown(f"**Computed in code (deterministic, tested):** {deterministic}  \n"
                    f"**LLM:** {llm}  \n"
                    "**Spec (source of truth):** " + ", ".join(f"[`specs/{s}`]({spec_url(s)})" for s in specs))
        for s in specs:
            st.divider()
            st.markdown((config.SPECS_DIR / s).read_text())


def store_health(store: Store) -> None:
    bad = store.unreadable()
    if bad:
        st.warning("Some saved records were written by an older version of GIOS and are ignored: "
                   + ", ".join(f"{n} {t}" for t, n in bad.items()) + ".", icon=":material/history:")
        if st.button("Reset this workspace", key="gios_reset_store"):
            store.clear()
            st.rerun()


def page_header(module: ModuleInfo) -> Store:
    st.set_page_config(page_title=f"{module.name} · GIOS", layout="wide")
    if config.is_demo_mode():
        st.info("**Demo mode.** LLM steps return cached outputs from `demo/`. Every number on this "
                "page is computed live in Python.", icon="🧪")
    st.title(module.name)
    st.caption(f"{module.purpose}  ·  Layer: {module.layer.title()}")
    how_it_works("this module", module.deterministic, module.llm, [module.spec])
    store = get_store()
    store_health(store)
    return store


def markdown_report(markdown: str, filename: str) -> None:
    st.download_button("Download as Markdown", markdown, file_name=filename, mime="text/markdown",
                       icon=":material/download:")
    st.markdown(markdown)


def save_signals(signals: list[Signal], module: str, key: str, store: Optional[Store] = None) -> None:
    col1, col2 = st.columns([1, 3])
    if col1.button(f"Save {len(signals)} signals to GIOS store", key=key, type="primary"):
        (store or get_store()).replace_module_output(signals, module=module)
        col2.success(f"Saved {len(signals)} signals (replacing this module's previous output).")


def seed_button(step: str, label: str, key: str, store: Store) -> None:
    """Offer to run every upstream module with defaults, then rerun the page."""
    from gios.modules.demo_flow import seed_through

    if st.button(label, key=key, type="primary"):
        with st.spinner("Running the upstream modules with default settings…"):
            seed_through(step, store)
        st.rerun()


def page_link(container, page: str, label: str, icon: str) -> None:
    """st.page_link that degrades to plain text when the page runs outside st.navigation (tests)."""
    from streamlit.errors import StreamlitAPIException

    try:
        container.page_link(page, label=label, icon=icon)
    except StreamlitAPIException:
        container.markdown(f"{label} (`{page}`)")
