import pandas as pd
import streamlit as st

from gios.core.schemas import HYPOTHESIS_PARTS, Hypothesis, Signal
from gios.core.store import Store
from gios.modules import MODULES
from gios.modules.hypothesis_evidence_validator import MODULE, analysis, pipeline
from gios.modules.hypothesis_evidence_validator.analysis import BAND_LABELS, RECOMMENDATION_LABELS
from gios.modules.hypothesis_evidence_validator.examples import ASSUMPTION_ONLY
from gios.modules.hypothesis_evidence_validator.models import DIMENSION_LABELS, DIMENSIONS, RUBRIC
from gios.modules.hypothesis_evidence_validator.report import to_markdown
from gios.ui import page_header

INFO = next(m for m in MODULES if m.slug == MODULE)
page_header(INFO)
store = Store()
signals = store.list(Signal)
stored = store.list(Hypothesis)

# --- Input -----------------------------------------------------------------------------------
mode = st.radio("Hypothesis", ["Stored hypothesis", "Enter manually"], horizontal=True, key="hev_mode",
                index=0 if stored else 1)
if mode == "Stored hypothesis":
    if not stored:
        st.info("No stored hypotheses yet. Run the **Growth Intelligence Diagnostic** and save its hypotheses, "
                "or enter one manually.")
        st.stop()
    options = {h.id: h for h in stored}
    hid = st.selectbox("Choose a hypothesis", list(options), key="hev_pick",
                       format_func=lambda i: f"{options[i].label}  ·  {i}")
    hypothesis = options[hid]
else:
    for part in ("title",) + HYPOTHESIS_PARTS:
        st.session_state.setdefault(f"hev_{part}", getattr(ASSUMPTION_ONLY, part))
    st.caption("Prefilled with a plausible but assumption-only example. Edit any part or link signals from the store.")
    title = st.text_input("Title", key="hev_title")
    cols = st.columns(2)
    parts = {p: cols[i % 2].text_area(analysis.PART_LABELS[p], key=f"hev_{p}", height=80)
             for i, p in enumerate(HYPOTHESIS_PARTS)}
    by_id = {s.id: s for s in signals}
    linked = st.multiselect("Linked signals (supporting)", list(by_id), key="hev_signals",
                            format_func=lambda i: f"{i} · {by_id[i].type}")
    hypothesis = Hypothesis(title=title, signal_ids=linked, **parts)
    hypothesis.id = "hev-" + hypothesis.fingerprint()
    if hypothesis.fingerprint() == ASSUMPTION_ONLY.fingerprint():
        hypothesis.id = ASSUMPTION_ONLY.id

# --- Standard check --------------------------------------------------------------------------
st.subheader("Hypothesis standard")
standard = analysis.check_standard(hypothesis)
cols = st.columns(3)
for i, c in enumerate(standard):
    icon = {"ok": ":material/check_circle:", "missing": ":material/cancel:", "vague": ":material/warning:"}[c.status]
    color = {"ok": "green", "missing": "red", "vague": "orange"}[c.status]
    cols[i % 3].markdown(f":{color}[{icon} **{c.label}**]  \n{c.text or '_missing_'}")
bad = [c.label for c in standard if not c.ok]
if bad:
    st.error("Incomplete hypothesis. Missing or vague: " + ", ".join(bad), icon=":material/rule:")

# --- Scoring -----------------------------------------------------------------------------------
fp = hypothesis.fingerprint()
cache = st.session_state.setdefault("hev_proposals", {})
if fp not in cache:
    em = analysis.evidence_map(hypothesis, signals)
    with st.spinner("Proposing scores…"):
        cache[fp] = pipeline.propose(hypothesis, em, standard)
proposal, error = cache[fp]
if error:
    st.warning(f"No LLM proposal available ({error}). Scores start at 0; set them below.", icon=":material/edit:")

base = pipeline.validate(hypothesis, signals, proposal=proposal, auto_propose=False, proposal_error=error)
st.subheader("Score")
st.caption("Proposed by the LLM, capped by code to what the linked evidence supports. Edit **Score** to override.")
table = pd.DataFrame([{
    "dimension": r.dimension, "Dimension": DIMENSION_LABELS[r.dimension],
    "Proposed": r.proposed, "Evidence cap": r.cap, "Score": r.score, "Justification": r.justification,
    "Cap reason": r.cap_reason or "",
} for r in base.rows])
edited = st.data_editor(
    table, hide_index=True, width="stretch", key=f"hev_scores_{fp}",
    disabled=["Dimension", "Proposed", "Evidence cap", "Justification", "Cap reason"],
    column_config={"dimension": None,
                   "Score": st.column_config.NumberColumn(min_value=0, max_value=2, step=1, required=True)},
)
overrides = {r.dimension: int(r.Score) for r in edited.itertuples()}
result = pipeline.validate(hypothesis, signals, overrides=overrides, proposal=proposal, auto_propose=False,
                           proposal_error=error)
above = [DIMENSION_LABELS[r.dimension] for r in result.rows if r.above_cap]
if above:
    st.warning("Overridden above the evidence cap: " + ", ".join(above) + ". The report will say so.")

c1, c2, c3 = st.columns(3)
c1.metric("Total", f"{result.total} / 12")
c2.metric("Band", BAND_LABELS[result.band])
c3.metric("Recommendation", RECOMMENDATION_LABELS[result.recommendation])

st.divider()
c1, c2 = st.columns([1, 3])
if c1.button("Save score to hypothesis", type="primary", key="hev_save"):
    pipeline.save(result, store)
    c2.success(f"Saved scores to `{hypothesis.id}`.")
markdown = to_markdown(result)
st.download_button("Download as Markdown", markdown, file_name=f"hypothesis_validation_{hypothesis.id}.md",
                   mime="text/markdown", icon=":material/download:")
st.markdown(markdown)
