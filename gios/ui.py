"""Shared Streamlit UI helpers for module pages."""
from __future__ import annotations

import re

import streamlit as st

from gios import config
from gios.core.schemas import Signal
from gios.core.store import Store
from gios.modules import ModuleInfo

# Reference palette (dataviz skill): categorical slot 1 and slot 2, plus a recessive neutral.
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"
NEUTRAL = "#a3a29c"


def page_header(module: ModuleInfo) -> None:
    st.set_page_config(page_title=f"{module.name} · GIOS", layout="wide")
    if config.is_demo_mode():
        st.info("**Demo mode.** LLM steps return cached outputs from `demo/`. Every number on this "
                "page is computed live in Python.", icon="🧪")
    st.title(module.name)
    st.caption(f"{module.purpose}  ·  Layer: {module.layer.title()}  ·  Spec: `specs/{module.spec}`")
    with st.expander("Module spec (source of truth)"):
        st.markdown(module.spec_path.read_text())


def markdown_report(markdown: str, filename: str) -> None:
    st.download_button("Download as Markdown", markdown, file_name=filename, mime="text/markdown",
                       icon=":material/download:")
    # Streamlit renders "###" headings; keep the report exactly as downloaded.
    st.markdown(markdown)


def save_signals(signals: list[Signal], module: str, key: str) -> None:
    col1, col2 = st.columns([1, 3])
    if col1.button(f"Save {len(signals)} signals to GIOS store", key=key, type="primary"):
        Store().replace_module_output(signals, module=module)
        col2.success(f"Saved {len(signals)} signals (replacing this module's previous output).")


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
