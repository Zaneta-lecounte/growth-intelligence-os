import math
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pydantic import ValidationError

from gios.core import data
from gios.core.schemas import Variant
from gios.modules import MODULES
from gios.modules.downstream_impact_analyzer import MODULE, Settings, load, pipeline
from gios.modules.downstream_impact_analyzer.analysis import INTERPRETATION_LABELS, RECOMMENDATION_LABELS
from gios.modules.downstream_impact_analyzer.examples import VOLUME_QUALITY_EXAMPLE
from gios.modules.downstream_impact_analyzer.report import to_markdown
from gios.ui import NEUTRAL, SERIES_1, SERIES_2, page_header

INFO = next(m for m in MODULES if m.slug == MODULE)
store = page_header(INFO)

with st.sidebar:
    st.header("Test settings")
    alpha = st.select_slider("Significance level α", [0.01, 0.05, 0.10], value=0.05)
    min_events = st.number_input("Underpowered below (events per arm)", 5, 500, Settings.min_events, 5)
settings = Settings(alpha=alpha, min_events=int(min_events))

source = st.radio("Experiment", ["From experiments.csv", "Manual entry"], horizontal=True, key="dia_source")
if source == "From experiments.csv":
    exps = data.load("experiments").drop_duplicates("experiment_id").set_index("experiment_id")
    eid = st.selectbox("Choose an experiment", list(exps.index), key="dia_pick",
                       format_func=lambda i: f"{i} · {exps.loc[i, 'experiment_name']}")
    loaded = load(eid)
    variants = loaded["variants"]
    st.caption(f"Page: {loaded['page']} · {loaded['start']} to {loaded['end']}, outcomes observed through "
               f"{loaded['observed_through']}. Hypothesis: {loaded['hypothesis']}")
    result = pipeline.run(eid, settings)
else:
    ex = VOLUME_QUALITY_EXAMPLE
    c1, c2, c3, c4 = st.columns(4)
    name = c1.text_input("Experiment name", ex["name"], key="dia_name")
    start = c2.date_input("Start", ex["start"], key="dia_start")
    observed = c3.date_input("Outcomes observed through", ex["observed_through"], key="dia_obs")
    velocity_default = float(data.load("lead_velocity").median_days_lead_to_win.median())
    velocity = c4.number_input("Median lead→win days", 1.0, 720.0, velocity_default, 1.0, key="dia_velocity",
                               help="Defaults to the median across sources and segments in lead_velocity.csv")
    st.caption("Control first. Prefilled with a synthetic test where leads rise but the SQL rate falls.")
    variants = st.data_editor(ex["variants"], key="dia_variants", width="stretch",
                              column_config={c: st.column_config.NumberColumn(min_value=0, step=1)
                                             for c in ex["variants"].columns})
    try:
        for v, r in variants.iterrows():
            Variant(name=str(v), **{k: r[k] for k in variants.columns})
    except ValidationError as exc:
        st.error(f"Check the variant counts: {exc.errors()[0]['msg']}")
        st.stop()
    result = pipeline.run_manual(name, variants, start, observed, velocity, settings)

c1, c2, c3 = st.columns([2, 1, 1])
c1.metric("Business interpretation", INTERPRETATION_LABELS[result.interpretation])
c2.metric("Recommendation", RECOMMENDATION_LABELS[result.recommendation])
p = result.stage("conversions").test
c3.metric("Primary lift", f"{p['lift']:+.1%}", "significant" if p["significant"] else "not significant",
          delta_color="off")

rows = [(s.label, s) for s in result.stages] + [(f"{q.label} (quality)", q) for q in result.quality]
fig = go.Figure()
for label, s in rows:
    t = s.test
    base = t["p_control"] or math.nan
    lift, lo, hi = t["lift"], t["ci_low"] / base, t["ci_high"] / base
    color = (SERIES_2 if t["diff"] < 0 else SERIES_1) if t["significant"] else NEUTRAL
    fig.add_trace(go.Scatter(
        x=[lift], y=[label], mode="markers", marker=dict(size=11, color=color, line=dict(width=2, color="white")),
        error_x=dict(type="data", symmetric=False, array=[hi - lift], arrayminus=[lift - lo], thickness=2, width=0,
                     color=color),
        showlegend=False, hovertemplate=f"{label}: %{{x:+.1%}} [{lo:+.1%}, {hi:+.1%}]"
                                        + (" · underpowered" if s.underpowered else "") + "<extra></extra>"))
fig.add_vline(x=0, line_color=NEUTRAL, line_width=1)
fig.update_yaxes(categoryorder="array", categoryarray=[r[0] for r in rows][::-1], title=None)
fig.update_xaxes(tickformat="+.0%", title="Relative lift vs control (95% CI)")
fig.update_layout(height=380, margin=dict(l=0, r=10, t=10, b=0))
st.plotly_chart(fig, width="stretch")
st.caption("Blue = significantly better, orange = significantly worse, grey = not significant.")

st.divider()
c1, c2 = st.columns([1, 3])
if c1.button("Save result to experiment", type="primary", key="dia_save"):
    pipeline.save(result, variants, store)
    c2.success(f"Saved `{result.experiment_id}` with its interpretation. Capture the learning next.")
markdown = to_markdown(result)
st.download_button("Download as Markdown", markdown, file_name=f"downstream_impact_{result.experiment_id}.md",
                   mime="text/markdown", icon=":material/download:")
st.markdown(markdown)
