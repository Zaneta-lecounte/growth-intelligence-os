import hashlib
from typing import get_args

import pandas as pd
import streamlit as st

from gios.core.schemas import Dependency, Hypothesis, Opportunity, OpportunityCategory
from gios.modules import MODULES
from gios.modules.experiment_opportunity_scorer.analysis import RECOMMENDATION_LABELS
from gios.modules.growth_priority_orchestrator import MODULE, build, gather, propose, rebalance, save
from gios.modules.growth_priority_orchestrator.analysis import BUCKETS, HORIZONS, cluster_overlap
from gios.modules.growth_priority_orchestrator.report import to_markdown
from gios.ui import page_header, recommendation_badge, seed_button

INFO = next(m for m in MODULES if m.slug == MODULE)
store = page_header(INFO)

if not store.list(Opportunity) and not store.list(Hypothesis):
    st.warning("Nothing to orchestrate yet: the store has no opportunities or hypotheses.")
    seed_button("backlog", "Run every upstream module with defaults (signals → diagnostic → validator → scorer)", "gpo_seed", store)
    st.stop()

with st.sidebar:
    st.header("Capacity")
    capacity = st.number_input("Items the team can run Now", 1, 20, 3, key="gpo_capacity")
    if st.button("Reset roadmap decisions", key="gpo_reset", help="Clears saved merges, buckets and horizons."):
        for o in store.list(Opportunity):
            o.merged_into, o.merged_ids, o.horizon, o.horizon_order, o.portfolio_bucket = None, [], None, None, None
            store.save(o)
        for k in [k for k in st.session_state if k.startswith("gpo_") and k not in ("gpo_capacity",)]:
            del st.session_state[k]
        st.rerun()

inputs = gather(store)
st.caption(f"Pulled {len(inputs.candidates)} candidates (scored opportunities and unscored hypotheses), "
           f"{len(inputs.hypotheses)} hypotheses, {len(inputs.recommendations)} diagnostic recommendations and "
           f"{sum(inputs.signal_counts.values())} signals from the GIOS store.")

cand_key = hashlib.sha1(",".join(sorted(o.id for o in inputs.candidates)).encode()).hexdigest()[:8]
if st.session_state.get("gpo_proposal_key") != cand_key:
    with st.spinner("Looking for duplicates…"):
        st.session_state.gpo_proposal = propose(inputs.candidates, inputs.hypotheses)
    st.session_state.gpo_proposal_key = cand_key
    st.session_state.gpo_manual = []
proposal = st.session_state.gpo_proposal
titles = {o.id: o.title for o in inputs.candidates}

# --- Step 1: consolidate -----------------------------------------------------------------------
st.subheader("1 · Consolidate duplicates")
if proposal.error:
    st.warning(f"No LLM proposal ({proposal.error}). Merge duplicates manually below.")
if proposal.ignored_ids:
    st.caption("Ignored unknown ids in the proposal: " + ", ".join(proposal.ignored_ids))
merges = []
clusters = proposal.proposal.clusters if proposal.proposal else []
by_id = {o.id: o for o in inputs.candidates}
for i, c in enumerate(clusters):
    members = [by_id[m] for m in c.member_ids if not by_id[m].merged_into]
    if len(members) < 2:
        continue
    with st.container(border=True):
        confirmed = st.checkbox(f"Merge into **{c.canonical_title}**", key=f"gpo_merge_{i}")
        st.markdown("\n".join(f"- {o.title} (`{o.id}`, score {o.priority_score:,.0f})" for o in members))
        st.caption(f"{c.rationale} Linked-signal overlap: {cluster_overlap(members):.0%}.")
    if confirmed:
        merges.append(([o.id for o in members], c.canonical_title))
with st.expander("Merge manually"):
    pick = st.multiselect("Items that are the same opportunity", list(titles), format_func=titles.get, key="gpo_pick")
    name = st.text_input("Title for the merged item (optional)", key="gpo_pick_title")
    if st.button("Add merge", key="gpo_add_merge", disabled=len(pick) < 2):
        st.session_state.gpo_manual.append((list(pick), name or None))
merges += st.session_state.gpo_manual

roadmap = build(inputs, proposal, merges, capacity)

# --- Step 2–3: classify, dependencies, bucket, horizon ------------------------------------------
st.subheader("2 · Classify, flag dependencies, and schedule")
st.caption("Defaults come from documented rules (bucket, horizon, minimum dependencies) and the LLM's dependency "
           "tags. Edit anything; order sets the position inside a horizon.")
rows = [{"id": o.id, "Title": o.title, "Score": o.priority_score,
         "Recommendation": RECOMMENDATION_LABELS[o.final_recommendation],
         "Category": o.category, "Bucket": o.portfolio_bucket, "Dependencies": o.dependencies,
         "Horizon": o.horizon or "", "Order": o.horizon_order} for o in roadmap.active]
merge_key = hashlib.sha1(repr(sorted(merges)).encode()).hexdigest()[:8]
edited = st.data_editor(
    pd.DataFrame(rows), key=f"gpo_editor_{merge_key}", hide_index=True, width="stretch",
    disabled=["Title", "Score", "Recommendation"],
    column_config={
        "id": None,
        "Score": st.column_config.NumberColumn(format="%.0f"),
        "Category": st.column_config.SelectboxColumn(options=list(get_args(OpportunityCategory)), required=True),
        "Bucket": st.column_config.SelectboxColumn(options=list(BUCKETS), required=True),
        "Dependencies": st.column_config.MultiselectColumn(options=list(get_args(Dependency))),
        "Horizon": st.column_config.SelectboxColumn(options=["", *HORIZONS], help="Empty = off the roadmap"),
        "Order": st.column_config.NumberColumn(min_value=1, step=1),
    },
)
items = {o.id: o for o in roadmap.items}
for r in edited.to_dict("records"):
    o = items[r["id"]]
    o.category, o.portfolio_bucket = r["Category"], r["Bucket"]
    o.dependencies = list(r["Dependencies"] or [])
    o.horizon = r["Horizon"] or None
    o.horizon_order = int(r["Order"]) if pd.notna(r["Order"]) else None
roadmap.items = list(items.values())
roadmap = rebalance(roadmap)

# --- Step 4–5: balance --------------------------------------------------------------------------
st.subheader("3 · Portfolio balance")
cols = st.columns(len(BUCKETS))
for col, (b, label) in zip(cols, BUCKETS.items()):
    n = roadmap.balance.counts[b]
    col.metric(f"{label} ({n} item{'s' if n != 1 else ''})", f"{roadmap.balance.shares[b]:.0%}")
for w in roadmap.balance.warnings:
    st.warning(w, icon=":material/warning:")
for n in roadmap.balance.notes:
    st.caption(n)
if not roadmap.balance.warnings:
    st.success("Balanced: no bucket above 50%, not all top-funnel, Now within capacity.")

# --- Step 6: board ------------------------------------------------------------------------------
st.subheader("4 · Now / Next / Later")
for col, (h, label) in zip(st.columns(3), HORIZONS.items()):
    col.markdown(f"**{label}**")
    for o in roadmap.lane(h):
        with col.container(border=True):
            rec = o.final_recommendation
            st.markdown(f"**{o.horizon_order}. {o.title}**  \n"
                        f"{recommendation_badge(rec, RECOMMENDATION_LABELS[rec])} :blue-badge[{BUCKETS[o.portfolio_bucket]}]"
                        f"  \nscore {o.priority_score:,.0f} · :gray[{', '.join(o.dependencies) or 'no dependencies'}]")

st.divider()
c1, c2 = st.columns([1, 3])
if c1.button("Save roadmap", type="primary", key="gpo_save"):
    save(roadmap, store)
    c2.success("Saved merges, buckets, dependencies and horizons on the Opportunity records.")
markdown = to_markdown(roadmap)
st.download_button("Download as Markdown", markdown, file_name="growth_priority_orchestrator.md",
                   mime="text/markdown", icon=":material/download:")
st.markdown(markdown)
