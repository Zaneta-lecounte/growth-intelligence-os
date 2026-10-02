import streamlit as st

from gios.core.llm import LLMError
from gios.core.schemas import Experiment, Learning
from gios.modules import MODULES
from gios.modules.customer_signal_synthesizer.models import THEME_LABELS
from gios.modules.downstream_impact_analyzer import analyze_all
from gios.modules.downstream_impact_analyzer.analysis import INTERPRETATION_LABELS
from gios.modules.experiment_learning_capture import MODULE, pipeline
from gios.modules.experiment_learning_capture.report import DECISION_LABELS, NEXT_PARTS, to_markdown
from gios.ui import page_header

INFO = next(m for m in MODULES if m.slug == MODULE)
store = page_header(INFO)

experiments = {e.id: e for e in store.list(Experiment)}
if not experiments:
    st.warning("No analyzed experiments in the store yet.")
    if st.button("Analyze all past experiments with the Downstream Impact Analyzer", key="elc_seed", type="primary"):
        analyze_all(store)
        st.rerun()
    st.stop()

eid = st.selectbox("Experiment", list(experiments), key="elc_pick",
                   format_func=lambda i: f"{i} · {experiments[i].name} · "
                                         f"{INTERPRETATION_LABELS.get(experiments[i].interpretation or '', 'not analyzed')}")
e = experiments[eid]
saved = store.get(Learning, pipeline.learning_id(eid))

# Draft once per experiment: a saved learning wins over a fresh LLM draft.
key = f"elc_draft_{eid}"
if key not in st.session_state:
    if saved:
        st.session_state[key] = (saved, "")
    else:
        try:
            with st.spinner("Drafting the learning…"):
                st.session_state[key] = (pipeline.build(e, pipeline.draft(e)), "")
        except LLMError as exc:
            st.session_state[key] = (pipeline.build(e, None), str(exc))
base, error = st.session_state[key]
if error:
    st.warning(f"No LLM draft available ({error}). Fill in the sections yourself.")
elif saved:
    st.caption("Loaded the saved learning from the Experiment Library.")

st.subheader("Computed by the Downstream Impact Analyzer")
st.markdown(f"**Statistical / directional result.** {base.statistical_result}  \n"
            f"**Segment findings.** {base.segment_findings}")

st.subheader("Learning (drafted by the LLM, yours to edit)")
f = st.form(f"elc_form_{eid}")
labels = {"original_problem": "Original problem", "what_happened": "What happened",
          "learned_about_customer": "What we learned about the customer",
          "learned_about_journey": "What we learned about the journey",
          "learned_about_business": "What we learned about the business",
          "should_not_conclude": "What we should NOT conclude (required)", "reusable_principle": "Reusable principle"}
edits = {k: f.text_area(label, getattr(base, k), key=f"elc_{eid}_{k}", height=90) for k, label in labels.items()}
c1, c2, c3 = f.columns(3)
decisions = list(DECISION_LABELS)
edits["decision"] = c1.selectbox("Decision", decisions, index=decisions.index(base.decision),
                                 format_func=DECISION_LABELS.get, key=f"elc_{eid}_decision")
themes = list(THEME_LABELS)
edits["theme"] = c2.selectbox("Theme", themes, index=themes.index(base.theme) if base.theme in themes else 0,
                              format_func=THEME_LABELS.get, key=f"elc_{eid}_theme")
edits["segment"] = c3.text_input("Segment", base.segment, key=f"elc_{eid}_segment")
f.markdown("**Next hypothesis**")
parts = dict(base.next_hypothesis_parts)
new_parts = {"title": f.text_input("Title", parts.get("title", base.next_hypothesis), key=f"elc_{eid}_nh_title")}
for k, label in NEXT_PARTS:
    new_parts[k] = f.text_input(label, parts.get(k, ""), key=f"elc_{eid}_nh_{k}")
submitted = f.form_submit_button("Save to Experiment Library", type="primary")

learning = base.model_copy(update=edits | {"next_hypothesis_parts": new_parts,
                                           "next_hypothesis": f"{new_parts['title']}: {new_parts['intervention']}"})
problems = pipeline.check(learning)
if problems:
    st.error(" ".join(problems) + " Saving is blocked until it is filled in.", icon=":material/block:")
if submitted and not problems:
    pipeline.save(learning, store)
    st.session_state[key] = (learning, "")
    st.success("Saved to the Experiment Library.")

if store.get(Learning, learning.id):
    if st.button("Send next hypothesis to the Validator", key="elc_send"):
        h = pipeline.send_next_hypothesis(store.get(Learning, learning.id), store)
        st.session_state.hev_mode = "Stored hypothesis"
        st.session_state.hev_pick = h.id
        st.switch_page("views/5_Hypothesis_Evidence_Validator.py")

st.divider()
markdown = to_markdown(learning)
st.download_button("Download as Markdown", markdown, file_name=f"learning_{eid}.md", mime="text/markdown",
                   icon=":material/download:")
st.markdown(markdown)
