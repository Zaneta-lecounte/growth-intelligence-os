import uuid
from typing import get_args

import pandas as pd
import plotly.express as px
import streamlit as st
from pydantic import ValidationError

from gios.core.schemas import Hypothesis, Opportunity, OpportunityCategory
from gios.modules import MODULES
from gios.modules.experiment_opportunity_scorer import MODULE, Settings, analysis, build_backlog, save, score
from gios.modules.experiment_opportunity_scorer.report import JUDGMENT_NOTE, to_markdown
from gios.ui import NEUTRAL, SERIES_1, page_header, seed_button

INFO = next(m for m in MODULES if m.slug == MODULE)
store = page_header(INFO)

if not store.list(Hypothesis) and not store.list(Opportunity):
    st.warning("No hypotheses or opportunities in the store yet.")
    seed_button("validations", "Run the upstream modules with defaults (signals → diagnostic → validator)",
                "eos_seed", store)
    st.stop()

with st.sidebar:
    st.header("Rules")
    defer = st.number_input("Defer below priority score", 0.0, 3125.0, Settings.defer_threshold, 5.0)
    weeks = st.number_input("Sample must be reachable within (weeks)", 1.0, 52.0, Settings.max_weeks, 1.0)
settings = Settings(defer_threshold=defer, max_weeks=weeks)

st.info(f"**{JUDGMENT_NOTE}**", icon=":material/balance:")

if "eos_backlog" not in st.session_state or st.button("Reload backlog from store", key="eos_reload"):
    st.session_state.eos_backlog = build_backlog(store)
    st.session_state.pop("eos_editor", None)
base = {o.id: o for o in st.session_state.eos_backlog}

# --- Editable backlog --------------------------------------------------------------------------
st.subheader("Backlog")
st.caption("Edit any GROWTH score, flag risks, or add a row at the bottom. G and T default to 3 and need your "
           "judgment. Sample-size risk is computed (not editable).")
categories = list(get_args(OpportunityCategory))
recs = [None] + list(analysis.RECOMMENDATION_LABELS)
rows = []
for o in base.values():
    row = {"id": o.id, "Title": o.title, "Category": o.category, "MDE": o.mde}
    row |= {analysis.GROWTH_LETTERS[g]: getattr(o, g) for g in analysis.GROWTH}
    row |= {analysis.RISK_LABELS[f]: f in o.risk_flags for f in analysis.RISK_LABELS if f != "sample_size"}
    row |= {"Override": o.recommendation_override, "Override reason": o.override_reason}
    rows.append(row)
frame = pd.DataFrame(rows)
score_col = lambda label: st.column_config.NumberColumn(label, min_value=1, max_value=5, step=1, required=True,  # noqa: E731
                                                          default=3)
edited = st.data_editor(
    frame, key="eos_editor", num_rows="dynamic", hide_index=True, width="stretch",
    column_config={
        "id": None,
        "Title": st.column_config.TextColumn(required=True, width="large"),
        "Category": st.column_config.SelectboxColumn(options=categories, required=True, default="customer_problem"),
        "G": score_col("G"), "R": score_col("R"), "O": score_col("O"), "W": score_col("W"),
        "T": score_col("T (effort)"), "H": score_col("H"),
        "MDE": st.column_config.NumberColumn("MDE (relative)", min_value=0.01, max_value=5.0, step=0.05,
                                             format="percent", default=0.2),
        "Override": st.column_config.SelectboxColumn(options=recs, help="Requires a reason"),
    },
)

items, problems = [], []
for r in edited.to_dict("records"):
    if not str(r.get("Title") or "").strip():
        continue
    oid = r.get("id") if isinstance(r.get("id"), str) else f"eos-manual-{uuid.uuid4().hex[:8]}"
    o = base.get(oid) or Opportunity(id=oid, title=r["Title"], category=r.get("Category") or "customer_problem")
    update = {"title": r["Title"], "category": r.get("Category") or o.category, "mde": float(r.get("MDE") or 0.2)}
    update |= {g: int(r.get(analysis.GROWTH_LETTERS[g]) or 3) for g in analysis.GROWTH}
    update["risk_flags"] = [f for f, label in analysis.RISK_LABELS.items() if f != "sample_size" and r.get(label)]
    override = r.get("Override") if isinstance(r.get("Override"), str) else None
    reason = r.get("Override reason") if isinstance(r.get("Override reason"), str) else ""
    try:
        o = Opportunity(**(o.model_dump() | update | {"recommendation_override": override, "override_reason": reason}))
    except ValidationError:
        problems.append(r["Title"])
        o = Opportunity(**(o.model_dump() | update | {"recommendation_override": None}))
    items.append(o)
if problems:
    st.error("An override needs a reason (at least a few words). Ignored for: " + "; ".join(problems),
             icon=":material/edit_note:")

sb = score(items, settings, store)

# --- Ranked view -----------------------------------------------------------------------------
st.subheader("Ranked")
ranked = sb.ranked()
table = pd.DataFrame([{
    "Rank": i, "Opportunity": o.title, "Score": o.priority_score,
    "Recommendation": analysis.RECOMMENDATION_LABELS[o.final_recommendation]
    + (" (override)" if o.recommendation_override else ""),
    "Key risk": o.key_risk,
} for i, o in enumerate(ranked, 1)])
st.dataframe(table, hide_index=True, width="stretch",
             column_config={"Score": st.column_config.NumberColumn(format="%.1f")})
chart = table.iloc[::-1].assign(run=lambda d: d.Recommendation.str.startswith("Run now"),
                                label=lambda d: d.Rank.astype(str) + ". " + d.Opportunity.str.slice(0, 60))
fig = px.bar(chart, x="Score", y="label", orientation="h", log_x=True, color="run",
             color_discrete_map={True: SERIES_1, False: NEUTRAL}, text=chart.Score.map(lambda v: f"{v:,.1f}"),
             hover_data={"Recommendation": True, "run": False, "label": False})
fig.update_traces(textposition="outside", cliponaxis=False, marker_cornerradius=4)
fig.update_xaxes(range=[-0.7, 3.6], title="Priority score (log scale; possible range 0.2 to 3,125)")
fig.update_yaxes(title=None)
fig.update_layout(showlegend=False, height=120 + 36 * len(chart), margin=dict(l=0, r=30, t=10, b=0))
st.plotly_chart(fig, width="stretch")
st.caption("Blue = Run now. Grey = any other recommendation.")

with st.expander("How the defaults and flags were computed"):
    for o in ranked:
        notes = sb.notes.get(o.id, {})
        st.markdown(f"**{o.title}**  \n"
                    + "  \n".join(f"{analysis.GROWTH_LETTERS[g]}: {notes[g]}" for g in analysis.GROWTH if g in notes)
                    + f"  \nSample size: {sb.checks[o.id].describe(settings.max_weeks)}")

st.divider()
c1, c2 = st.columns([1, 3])
if c1.button(f"Save {len(sb.items)} opportunities", type="primary", key="eos_save"):
    save(sb, store)
    st.session_state.eos_backlog = build_backlog(store)
    c2.success("Saved as Opportunity records. Plan the roadmap in the Growth Priority Orchestrator.")
markdown = to_markdown(sb)
st.download_button("Download as Markdown", markdown, file_name="experiment_opportunity_scorer.md",
                   mime="text/markdown", icon=":material/download:")
st.markdown(markdown)
