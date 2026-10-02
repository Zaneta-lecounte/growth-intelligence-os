import streamlit as st

from gios.core import data
from gios.core.llm import LLMError
from gios.core.schemas import Learning, Signal
from gios.core.store import Store
from gios.modules import MODULES
from gios.modules.demo_flow import seed_through
from gios.modules.growth_council_insight_brief import MODULE, gather, slack, write
from gios.modules.growth_council_insight_brief.report import to_markdown
from gios.modules.qualified_demand_leakage_auditor.analysis import OWNERS
from gios.ui import page_header

INFO = next(m for m in MODULES if m.slug == MODULE)
page_header(INFO)
store = Store()

if not store.list(Signal) or not store.list(Learning):
    st.warning("The brief draws on every layer, and the store is missing signals or learnings.")
    if st.button("Run every module with defaults (signals → … → roadmap → experiments → learnings)",
                 key="gcb_seed", type="primary"):
        with st.spinner("Seeding the store…"):
            seed_through("learnings", store)
        st.rerun()
    st.stop()

months = sorted(data.load("funnel_by_source").month.unique())
start, end = st.select_slider("Period", options=months, value=(months[0], months[-1]), key="gcb_period")
inputs = gather(store, start, end)
c = inputs.counts
st.caption(f"In scope: {c['signals']} signals, {c['experiments']} experiments that ended in the period with "
           f"{c['learnings']} learnings, plus the current {c['hypotheses']} hypotheses and {c['roadmap']} roadmap items.")

key = f"gcb_brief_{start}_{end}"
if key not in st.session_state:
    try:
        with st.spinner("Writing the brief…"):
            st.session_state[key] = (write(inputs), "")
    except (LLMError, ValueError) as exc:
        st.session_state[key] = (None, str(exc))
brief, error = st.session_state[key]
if brief is None:
    st.error(f"No brief for this period: {error}", icon=":material/gpp_bad:")
    st.stop()

st.subheader("Owner")
c1, c2 = st.columns([1, 1])
options = list(OWNERS)
default = options.index(brief.owner)
brief.owner = c1.selectbox("Owner class (from the Leakage Auditor)", options, index=default,
                           format_func=OWNERS.get, key=f"{key}_owner")
brief.owner_name = c2.text_input("Named owner (team or person, optional)", key=f"{key}_owner_name")
if brief.suggestions:
    st.caption("Suggested by the cited evidence: " + ", ".join(OWNERS[s] for s in brief.suggestions))

markdown = to_markdown(brief)
summary = slack(brief)
st.subheader("Slack summary")
st.code(summary, language=None, wrap_lines=True)
c1, c2 = st.columns(2)
c1.download_button("Download brief (Markdown)", markdown, file_name=f"growth_council_brief_{start}_{end}.md",
                   mime="text/markdown", icon=":material/download:")
c2.download_button("Download Slack summary", summary, file_name=f"growth_council_slack_{start}_{end}.txt",
                   mime="text/plain", icon=":material/chat:")
st.divider()
st.markdown(markdown)
