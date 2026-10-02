import streamlit as st

from gios import config
from gios.modules import full_diagnostic
from gios.modules.experiment_opportunity_scorer.analysis import RECOMMENDATION_LABELS
from gios.modules.growth_priority_orchestrator.analysis import BUCKETS
from gios.modules.growth_priority_orchestrator.report import to_markdown
from gios.ui import get_store, how_it_works, page_link, recommendation_badge, store_health

st.set_page_config(page_title="Run Full Diagnostic · GIOS", layout="wide")
if config.is_demo_mode():
    st.info("**Demo mode.** LLM steps return cached outputs from `demo/`. Every number is computed live in Python.",
            icon="🧪")
st.title("Run Full Diagnostic")
st.caption("One click: Signal → Diagnosis → Validation → Scoring → Orchestration on EchoAI's synthetic data, "
           "then the Executive Growth Priorities.")
how_it_works(
    "the full diagnostic",
    "each module's deterministic core, in order, with default settings; proposed duplicate merges are confirmed "
    "automatically; the story and red-herring checks below match roadmap items to their linked evidence.",
    "the same LLM steps as the individual modules (cached in demo mode).",
    ["growth-intelligence-diagnostic.md", "experiment-opportunity-scorer.md", "growth-priority-orchestrator.md"],
)
store = get_store()
store_health(store)

if st.button("Run full diagnostic", type="primary", key="full_run", icon=":material/play_arrow:"):
    progress = st.progress(0.0, text="Starting…")
    with st.status("Running every layer…", expanded=True) as status:
        def on_step(i, key, label, n):
            detail = full_diagnostic.STEPS[i][2]
            st.write(f"✓ **{label}**: {detail} ({n} records)")
            progress.progress((i + 1) / len(full_diagnostic.STEPS), text=label)

        result = full_diagnostic.run(store, on_step)
        status.update(label="Full diagnostic complete", state="complete", expanded=False)
    st.session_state.full_result_ready = True
else:
    result = full_diagnostic.latest(store)

if result is None:
    st.info("Press **Run full diagnostic** to populate this workspace. Existing signals, hypotheses and "
            "opportunities in this workspace are replaced; experiments and learnings are kept.")
    st.stop()

# --- Story checks ---------------------------------------------------------------------------------
st.subheader("Did it find what matters?")
if result.passed:
    st.success("All three embedded stories are Now priorities, each linked to its full evidence, and the "
               "frequent-but-low-severity red herring is not on the roadmap.", icon=":material/verified:")
else:
    st.warning("Not every story check passed with the current settings. See below.")
cols = st.columns(len(result.stories) + 1)
for col, s in zip(cols, result.stories):
    with col.container(border=True):
        badge = ":green-badge[Now priority]" if s.prioritized else ":orange-badge[Not in Now]"
        st.markdown(f"**{s.name}** {badge}")
        st.caption(f"Roadmap item: {s.item.title if s.item else '—'}")
        for req in s.required:
            st.markdown(("✅ " if req in s.matched else "⬜ ") + req)
h = result.red_herring
with cols[-1].container(border=True):
    badge = ":green-badge[Kept off roadmap]" if h.deprioritized else ":red-badge[On roadmap]"
    st.markdown(f"**Red herring: cosmetic requests** {badge}")
    if h.signal:
        st.caption(f"Most frequent theme ({h.signal.attributes.get('verbatims')} verbatims, frequency rank "
                   f"{h.frequency_rank}) but ranked {h.overall_rank} of the customer themes on overall importance.")

# --- Executive priorities ----------------------------------------------------------------------------
st.subheader("Executive Growth Priorities")
now = result.roadmap.lane("now")
for i, o in enumerate(now[:3], 1):
    with st.container(border=True):
        rec = o.final_recommendation
        st.markdown(f"**{i}. {o.title}** {recommendation_badge(rec, RECOMMENDATION_LABELS[rec])} "
                    f":blue-badge[{BUCKETS[o.portfolio_bucket]}]")
        st.caption(f"Priority score {o.priority_score:,.0f} · {o.category.replace('_', ' ')} · dependencies: "
                   f"{', '.join(o.dependencies) or 'none'}" + (f" · {o.primary_metric}" if o.primary_metric else ""))
nxt = result.roadmap.lane("next")
if nxt:
    st.caption("Next: " + "; ".join(o.title for o in nxt))

st.markdown("Drill in: ")
c = st.columns(4)
page_link(c[0], "views/7_Growth_Priority_Orchestrator.py", "Roadmap", ":material/view_kanban:")
page_link(c[1], "views/6_Experiment_Opportunity_Scorer.py", "Scores", ":material/leaderboard:")
page_link(c[2], "views/4_Growth_Intelligence_Diagnostic.py", "Diagnosis", ":material/troubleshoot:")
page_link(c[3], "views/11_Growth_Council_Insight_Brief.py", "Council brief", ":material/groups:")

markdown = to_markdown(result.roadmap)
with st.expander("Full roadmap report"):
    st.download_button("Download as Markdown", markdown, file_name="gios_full_diagnostic.md", mime="text/markdown",
                       icon=":material/download:")
    st.markdown(markdown)
