"""GIOS home page: overview, layer diagram, module status, demo-mode banner."""
import json

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from gios import config
from gios.modules import LAYERS, MODULES
from gios.ui import page_link

st.set_page_config(page_title="GIOS · Growth Intelligence OS", layout="wide")

MERMAID = """
flowchart LR
    S["<b>1. Signal</b><br/>customer · behavioral · funnel<br/>acquisition · operational · revenue"]
    D["<b>2. Diagnosis</b><br/>where value leaks and why"]
    P["<b>3. Prioritization</b><br/>validate · score · sequence"]
    L["<b>4. Learning</b><br/>downstream impact · reusable knowledge"]
    S --> D --> P --> L
    L -. "learning loop" .-> S
"""


def render_mermaid(diagram: str, height: int = 260) -> None:
    """Render Mermaid via the CDN; falls back to showing the source if it can't load."""
    components.html(
        f"""
        <div id="diagram"><pre id="fallback" style="color:#666;font-size:12px"></pre></div>
        <script type="module">
          const src = {json.dumps(diagram)};
          document.getElementById("fallback").textContent = src;
          try {{
            const {{ default: mermaid }} = await import(
              "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs");
            mermaid.initialize({{ startOnLoad: false, theme: "neutral" }});
            const {{ svg }} = await mermaid.render("gios-layers", src);
            document.getElementById("diagram").innerHTML = svg;
          }} catch (err) {{
            console.warn("Mermaid unavailable; showing diagram source", err);
          }}
        </script>
        """,
        height=height,
    )

if config.is_demo_mode():
    st.info(
        "**Demo mode.** No `ANTHROPIC_API_KEY` is set, so LLM steps use cached outputs from "
        "`demo/`. All metrics and scores are computed live in Python either way.",
        icon="🧪",
    )

st.title("GIOS: Growth Intelligence Operating System")
st.markdown(
    """
Growth teams rarely lack data. What they lack is a shared way to connect **customer voice**,
**behavior**, **funnel quality**, and **experiments** into one decision. GIOS turns
fragmented signals into a diagnosis, a prioritized roadmap, and reusable learning.

The data here is synthetic. It describes **EchoAI**, a fictional B2B SaaS company selling AI meeting
transcription, over six months.

**Design rule:** all math, scoring, thresholds, and statistics are deterministic, tested Python.
The LLM only tags qualitative evidence and writes narrative. It never computes a number.
"""
)

page_link(st, "views/0_Run_Full_Diagnostic.py", "Run the full diagnostic in one click", ":material/play_circle:")

st.subheader("Four layers, one loop")
render_mermaid(MERMAID)

st.subheader("Modules")
status_label = {"planned": "⚪ Planned", "in_progress": "🟡 In progress", "ready": "🟢 Ready"}
table = pd.DataFrame(
    [
        {
            "Layer": LAYERS[m.layer],
            "Module": m.name,
            "Purpose": m.purpose,
            "Status": status_label[m.status],
            "Spec": f"specs/{m.spec}",
        }
        for m in MODULES
    ]
)
st.dataframe(table, hide_index=True, width="stretch")

with st.expander("Synthetic data"):
    files = sorted(config.DATA_DIR.glob("*.csv"))
    if files:
        st.write({f.name: f"{sum(1 for _ in f.open()) - 1:,} rows" for f in files})
    else:
        st.warning("No data yet. Run `python scripts/generate_data.py`.")

st.caption(f"Model: `{config.MODEL}` · Phase 0: foundation")
