import pandas as pd
import plotly.express as px
import streamlit as st

from gios.modules import MODULES
from gios.modules.customer_signal_synthesizer import MODULE, analysis, run
from gios.modules.customer_signal_synthesizer.report import to_markdown
from gios.ui import NEUTRAL, SERIES_1, markdown_report, page_header, save_signals

INFO = next(m for m in MODULES if m.slug == MODULE)
page_header(INFO)


@st.cache_data(show_spinner="Tagging customer evidence…")
def _run():
    return run()


result = _run()
st.markdown(
    f"**{result.meta['evidence_rows']}** evidence items · **{result.meta['unique_verbatims']}** distinct "
    f"verbatims tagged in batches · **{len(result.themes)}** themes. Frequency is counted from tags; "
    "severity and relevance are LLM-proposed. Edit them below and the ranking updates."
)

st.subheader("Score themes")
editable = result.themes[["theme", "label", "count", "frequency", *analysis.SCORE_COLUMNS]]
edited = st.data_editor(
    editable,
    hide_index=True,
    width="stretch",
    disabled=["theme", "label", "count", "frequency"],
    column_config={
        "theme": None,
        "label": "Theme",
        "count": st.column_config.NumberColumn("Verbatims", help="Counted in code from LLM tags"),
        "frequency": st.column_config.NumberColumn("Frequency (1–5)", help="Binned share of evidence"),
        **{c: st.column_config.NumberColumn(c.replace("_", " ").capitalize() + " (1–5)",
                                            min_value=1, max_value=5, step=1, required=True)
           for c in analysis.SCORE_COLUMNS},
    },
    key="css_scores",
)
scores = edited.set_index("theme")[analysis.SCORE_COLUMNS]
themes = result.themes.drop(columns=analysis.SCORE_COLUMNS).join(scores, on="theme")
themes = analysis.rank(themes)

left, right = st.columns(2)
with left:
    st.markdown("**Frequency is not importance**")
    chart = themes.assign(top=lambda d: d.overall_rank <= 3)
    fig = px.scatter(
        chart, x="count", y="severity_x_commercial", size="overall", text="label",
        color="top", color_discrete_map={True: SERIES_1, False: NEUTRAL},
        labels={"count": "Verbatims (frequency)", "severity_x_commercial": "Severity × commercial relevance"},
        hover_data={"overall": True, "frequency": True, "top": False, "label": False},
        size_max=36,
    )
    fig.update_traces(textposition="top center", marker=dict(line=dict(width=2, color="white")))
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0), height=380)
    st.plotly_chart(fig, width="stretch")
    st.caption("Blue = top 3 by overall score (F × S × C × J). Bubble size = overall score.")
with right:
    st.markdown("**Mentions per month, top themes**")
    top_themes = themes.head(4).theme.tolist()
    monthly = (result.tagged[result.tagged.theme.isin(top_themes)]
               .groupby(["month", "theme"]).size().unstack(fill_value=0))
    monthly.columns = [themes.set_index("theme").label[c] for c in monthly.columns]
    st.dataframe(monthly, width="stretch")

st.divider()
markdown = to_markdown(themes, result.synthesis, result.untagged, result.disagreements, result.meta)
signals = analysis.to_signals(themes, period=result.meta["period"])
save_signals(signals, MODULE, key="css_save")
markdown_report(markdown, "customer_signal_synthesizer.md")

with st.expander("Tagged evidence"):
    st.dataframe(
        result.tagged[["evidence_id", "date", "source", "segment", "journey_stage", "theme", "topic",
                       "sentiment", "business_relevance", "verbatim"]],
        hide_index=True, width="stretch",
    )
