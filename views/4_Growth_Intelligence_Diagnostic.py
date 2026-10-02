import streamlit as st

from gios.core import data
from gios.core.llm import LLMError
from gios.core.schemas import Signal
from gios.core.store import Store
from gios.modules import MODULES
from gios.modules.growth_intelligence_diagnostic import (
    MODULE,
    BusinessSignal,
    Filters,
    missing_business_fields,
)
from gios.modules.growth_intelligence_diagnostic import analysis, pipeline
from gios.modules.growth_intelligence_diagnostic.models import STEP1_FIELDS
from gios.modules.growth_intelligence_diagnostic.report import BADGE_UI, to_markdown
from gios.modules.signal_layer import run_signal_layer
from gios.ui import page_header

INFO = next(m for m in MODULES if m.slug == MODULE)
page_header(INFO)
store = Store()
funnel = data.load("funnel_by_source")
months = sorted(funnel.month.unique())
sources = sorted(funnel.source.unique())

all_signals = store.list(Signal)
if not all_signals:
    st.warning("The GIOS store has no signals yet. The diagnostic works from Phase 1 output.")
    if st.button("Run the signal layer with default settings and save its signals", key="gid_seed", type="primary"):
        with st.spinner("Running Customer Signal Synthesizer, Behavioral Friction Analyzer, Leakage Auditor…"):
            counts = run_signal_layer(store)
        st.success(f"Saved {sum(counts.values())} signals.")
        st.rerun()
    st.stop()

# --- Filters ---------------------------------------------------------------------------------
with st.sidebar:
    st.header("Scope")
    channel = st.selectbox("Acquisition source", ["All"] + sources, key="gid_channel",
                           format_func=lambda s: s.replace("_", " "))
    segment = st.selectbox("Segment", ["All", "smb", "mid_market"], key="gid_segment")
    start, end = st.select_slider("Period", options=months, value=(months[0], months[-1]), key="gid_period")
filters = Filters(channel=None if channel == "All" else channel, segment=None if segment == "All" else segment,
                  start=start, end=end)
in_scope = analysis.filter_signals(all_signals, filters.channel, filters.segment, start, end)

# --- Step 1: business signal -------------------------------------------------------------------
st.subheader("Step 1 · Define the business signal")
st.caption("Diagnosis is blocked until every field is filled in (spec: do not diagnose before the business "
           "signal is clear).")


def _fill_from_data():
    metric = st.session_state.gid_fill_metric
    draft = analysis.suggest_business_signal(funnel, metric, filters.channel, filters.segment, start, end)
    for k, v in draft.items():
        st.session_state[f"gid_bs_{k}"] = v
    if not st.session_state.get("gid_bs_goal"):
        st.session_state.gid_bs_goal = "Grow qualified pipeline efficiently"


c1, c2 = st.columns([2, 1])
c2.selectbox("Draft from data using", list(analysis.METRICS), key="gid_fill_metric")
c2.button("Fill from data", on_click=_fill_from_data, key="gid_fill", width="stretch",
          help="Computes the metric for this scope: later half of the period vs the earlier half.")
with c1:
    values = {}
    for name, label in STEP1_FIELDS.items():
        values[name] = st.text_input(label, key=f"gid_bs_{name}")
missing = missing_business_fields(values)
if missing:
    st.error("Diagnosis blocked. Still needed: " + ", ".join(missing), icon=":material/block:")

# --- Steps 2–4: signals and leakage point (deterministic) -----------------------------------
st.subheader("Step 2–3 · Signals in scope")
by_type = analysis.classify(in_scope)
cols = st.columns(len(by_type))
for col, (t, items) in zip(cols, by_type.items()):
    col.metric(t.title(), len(items))
with st.expander(f"All {len(in_scope)} signals, classified, with evidence status"):
    for t, items in by_type.items():
        if items:
            st.markdown(f"**{t.title()}**")
            for s in items:
                st.markdown(f"{BADGE_UI[s.evidence_status]} `{s.id}` · strength {s.strength} · {s.summary}")

st.subheader("Step 4 · Leakage point")
point = analysis.leakage_point(funnel, data.load("sales_feedback"), filters.channel, filters.segment, start, end)
if point:
    st.info(f"**{point.source} · {point.stage}** (from the Leakage Auditor's deterministic analysis). "
            + point.describe(), icon=":material/filter_alt:")
else:
    st.info("No stage is significantly below benchmark in this scope.")

# --- Steps 5–7: diagnosis --------------------------------------------------------------------
st.subheader("Steps 5–7 · Diagnose")
if st.button("Run diagnosis", type="primary", disabled=bool(missing) or not in_scope, key="gid_run"):
    try:
        with st.spinner("Generating root-cause hypotheses…"):
            st.session_state.gid_result = pipeline.diagnose(BusinessSignal(**values), filters, in_scope, point)
        st.session_state.pop("gid_error", None)
    except LLMError as exc:
        st.session_state.pop("gid_result", None)
        st.session_state.gid_error = str(exc)

if err := st.session_state.get("gid_error"):
    st.error(f"Diagnosis rejected: {err}", icon=":material/gpp_bad:")

result = st.session_state.get("gid_result")
if result is not None:
    if result.filters != filters:
        st.warning("Scope changed since this diagnosis ran. Run it again to refresh.")
    st.divider()
    c1, c2 = st.columns([1, 3])
    if c1.button(f"Save {len(result.ranked)} hypotheses to GIOS store", key="gid_save", type="primary"):
        pipeline.save(result, store)
        c2.success("Saved as Hypothesis records (plus the next best action as a Recommendation). Validate them in "
                   "the Hypothesis Evidence Validator.")
    st.download_button("Download as Markdown", to_markdown(result), file_name="growth_intelligence_diagnostic.md",
                       mime="text/markdown", icon=":material/download:")
    st.markdown(to_markdown(result, ui=True))
