import streamlit as st

from gios.core.schemas import Learning
from gios.core.store import Store
from gios.modules.customer_signal_synthesizer.models import THEME_LABELS
from gios.modules.downstream_impact_analyzer.analysis import INTERPRETATION_LABELS
from gios.modules.experiment_learning_capture import library
from gios.modules.experiment_learning_capture.report import DECISION_LABELS, to_markdown

st.set_page_config(page_title="Experiment Library · GIOS", layout="wide")
st.title("Experiment Library")
st.caption("Every captured learning, including losses and inconclusive results. Search across all sections.")

learnings = Store().list(Learning)
if not learnings:
    st.info("No learnings yet. Capture one in **Experiment Learning Capture**.")
    st.stop()

facets = library.facets(learnings)
query = st.text_input("Search", key="lib_query", placeholder="e.g. qualification, mobile, pricing")
c1, c2, c3, c4 = st.columns(4)
themes = c1.multiselect("Theme", facets["theme"], key="lib_theme", format_func=lambda t: THEME_LABELS.get(t, t))
pages = c2.multiselect("Page", facets["page"], key="lib_page")
segments = c3.multiselect("Segment", facets["segment"], key="lib_segment")
decisions = c4.multiselect("Decision", facets["decision"], key="lib_decision", format_func=DECISION_LABELS.get)
losses = st.toggle("Losses and inconclusive only", key="lib_losses")

results = library.search(learnings, query, themes, pages, segments, decisions, losses)
st.caption(f"{len(results)} of {len(learnings)} learnings")
for x in results:
    tag = INTERPRETATION_LABELS.get(x.interpretation or "", "")
    color = "red" if library.is_loss_or_inconclusive(x) else "green"
    with st.expander(f"**{x.experiment_name}** · {DECISION_LABELS[x.decision]} · {tag}"):
        st.markdown(f":{color}-badge[{tag or 'no interpretation'}] :blue-badge[{THEME_LABELS.get(x.theme, x.theme)}] "
                    f":gray-badge[{x.page}] :gray-badge[{x.segment}]")
        st.markdown(f"**Reusable principle:** {x.reusable_principle}  \n"
                    f"**Do NOT conclude:** {x.should_not_conclude}")
        md = to_markdown(x)
        st.download_button("Download as Markdown", md, file_name=f"learning_{x.experiment_id}.md",
                           mime="text/markdown", key=f"lib_dl_{x.id}")
        st.markdown(md)
