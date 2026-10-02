import plotly.graph_objects as go
import streamlit as st

from gios.modules import MODULES
from gios.modules.qualified_demand_leakage_auditor import MODULE, Settings, run, to_signals
from gios.modules.qualified_demand_leakage_auditor.analysis import OWNERS, STAGES
from gios.modules.qualified_demand_leakage_auditor.report import to_markdown
from gios.ui import NEUTRAL, SERIES_1, SERIES_2, markdown_report, page_header, save_signals

INFO = next(m for m in MODULES if m.slug == MODULE)
page_header(INFO)

with st.sidebar:
    st.header("Settings")
    min_volume = st.number_input("Minimum stage volume before flagging", 1, 5000, Settings.min_volume, 5)
    gap = st.slider("Sizing: share of the gap to benchmark closed", 0.05, 1.0, Settings.gap_closure, 0.05,
                    format="%.2f")
    sla = st.number_input("Follow-up SLA (hours)", 1.0, 168.0, Settings.sla_hours, 1.0)
    sla_share = st.slider("Flag SLA loss when this share of MQLs is late", 0.05, 0.8,
                          Settings.sla_share_threshold, 0.05, format="%.2f")
    qual = st.slider("Flag qualification mismatch at this share of rejections", 0.2, 0.95,
                     Settings.qualification_share_threshold, 0.05, format="%.2f")
settings = Settings(min_volume=int(min_volume), gap_closure=gap, sla_hours=sla, sla_share_threshold=sla_share,
                    qualification_share_threshold=qual)


@st.cache_data(show_spinner="Auditing the funnel…")
def _run(s: Settings):
    return run(settings=s)


result = _run(settings)
a = result.analysis
top = a.top_leak

if top is not None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Highest-value leak", f"{top.source} · {top.stage}")
    c2.metric("Conversion vs benchmark", f"{top.rate:.1%}", f"{top.rate - top.benchmark:+.1%} pts",
              delta_color="normal")
    c3.metric(f"Value of closing {settings.gap_closure:.0%} of the gap", f"${top.extra_revenue_per_month:,.0f}/mo",
              f"+{top.extra_wins_per_month:.1f} wins/mo", delta_color="off")

left, right = st.columns(2)
with left:
    stage_names = [name for _, _, name in STAGES] + ["Lead→Win"]
    default = stage_names.index(top.stage) if top is not None else 2
    stage = st.selectbox("Stage", stage_names, index=default)
    d = a.stage_map[a.stage_map.stage == stage].sort_values("rate")
    fig = go.Figure()
    for below, color, name in ((False, SERIES_1, "Conversion, in line or above (95% CI)"),
                               (True, SERIES_2, "Conversion, below benchmark (95% CI)")):
        part = d[(d.status == "below") == below]
        fig.add_trace(go.Scatter(
            x=part.rate, y=part.source, mode="markers", name=name,
            marker=dict(size=11, color=color, line=dict(width=2, color="white")),
            error_x=dict(type="data", symmetric=False, array=part.ci_high - part.rate,
                         arrayminus=part.rate - part.ci_low, thickness=2, width=0, color=color),
            hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=d.benchmark, y=d.source, mode="markers", marker=dict(symbol="line-ns", size=16, line=dict(width=2, color=NEUTRAL)),
        name="Benchmark (other sources)", hovertemplate="benchmark %{x:.1%}<extra></extra>"))
    fig.update_yaxes(categoryorder="array", categoryarray=d.source.tolist())
    fig.update_layout(xaxis_tickformat=".0%", height=320, margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")
    st.caption("Below benchmark = the whole 95% CI sits under the benchmark.")
with right:
    st.markdown("**Value of closing the gap, by leak**")
    leaks = a.leaks.head(8).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=leaks.extra_revenue_per_month, y=leaks.source + " · " + leaks.stage, orientation="h",
        marker=dict(color=[SERIES_1] * (len(leaks) - 1) + [SERIES_2] if len(leaks) else [], cornerradius=4),
        text=leaks.extra_revenue_per_month.map(lambda v: f"${v / 1000:,.0f}k"), textposition="outside",
        cliponaxis=False,
        customdata=leaks.owner.map(OWNERS),
        hovertemplate="%{y}<br>$%{x:,.0f}/month<br>Owner: %{customdata}<extra></extra>"))
    fig.update_layout(height=360, margin=dict(l=0, r=40, t=10, b=0), xaxis_title="Revenue per month",
                      xaxis_tickprefix="$", xaxis_range=[0, leaks.extra_revenue_per_month.max() * 1.2 if len(leaks) else 1])
    st.plotly_chart(fig, width="stretch")

st.divider()
signals = to_signals(result)
save_signals(signals, MODULE, key="qdl_save")
markdown_report(to_markdown(result), "qualified_demand_leakage_auditor.md")
