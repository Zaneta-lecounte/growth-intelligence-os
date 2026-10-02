import pandas as pd
import plotly.express as px
import streamlit as st

from gios.core.schemas import Signal
from gios.core.store import Store
from gios.modules import MODULES
from gios.modules.behavioral_friction_analyzer import MODULE, Thresholds, run, to_signals
from gios.modules.behavioral_friction_analyzer.report import to_markdown
from gios.ui import SERIES_1, markdown_report, page_header, save_signals

INFO = next(m for m in MODULES if m.slug == MODULE)
page_header(INFO)

with st.sidebar:
    st.header("Thresholds")
    z = st.slider("Minimum |z|", 2.0, 10.0, Thresholds.z, 0.5)
    dev = st.slider("Minimum deviation from baseline", 0.05, 0.60, Thresholds.min_deviation, 0.05, format="%.2f")
    vol = st.number_input("Minimum volume (sessions or form starts)", 50, 20000, Thresholds.min_volume, 50)
    load = st.slider("Load time: minimum slowdown vs site baseline", 0.10, 1.00, Thresholds.load_deviation, 0.05,
                     format="%.2f")
thresholds = Thresholds(z=z, min_deviation=dev, min_volume=int(vol), load_deviation=load)

customer = [s for s in Store().list(Signal) if s.type == "customer"]
if customer:
    st.success(f"Linking against {len(customer)} customer signals from the GIOS store.")
else:
    st.warning("No customer signals in the store, so no cause can be marked as supported. Run the "
               "**Customer Signal Synthesizer** and save its signals to link evidence.")


@st.cache_data(show_spinner="Detecting anomalies and classifying friction…")
def _run(th: Thresholds, customer_json: tuple[str, ...]):
    return run(thresholds=th, customer_signals=[Signal.model_validate_json(s) for s in customer_json])


result = _run(thresholds, tuple(s.model_dump_json() for s in customer))

if result.findings:
    rows = [{"Finding": f.title, "Flag": fl.label + (" (trend)" if fl.check == "trend" and "trend" not in fl.label else ""),
             "Deviation": fl.deviation, "z": fl.z}
            for f in result.findings for fl in f.flags]
    df = pd.DataFrame(rows)
    st.subheader(f"{len(result.findings)} findings")
    fig = px.bar(df, x="Deviation", y="Flag", facet_row="Finding", orientation="h",
                 color_discrete_sequence=[SERIES_1], text=df.Deviation.map(lambda d: f"{d:+.0%}"),
                 hover_data={"z": ":.1f", "Deviation": ":+.0%"})
    fig.update_yaxes(matches=None, title=None)
    fig.update_xaxes(tickformat="+.0%", title="Deviation from standardized baseline (bad direction)")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=", 1)[-1], textangle=0, x=0, xanchor="left",
                                               yanchor="bottom"))
    fig.update_layout(height=150 + 110 * len(result.findings), margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
    st.plotly_chart(fig, width="stretch")

st.divider()
signals = to_signals(result)
save_signals(signals, MODULE, key="bfa_save")
markdown_report(to_markdown(result), "behavioral_friction_analyzer.md")
