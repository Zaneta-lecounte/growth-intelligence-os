"""GIOS entry point: multipage navigation."""
import streamlit as st

PAGES = {
    "Overview": [st.Page("home.py", title="Home", icon=":material/home:", default=True)],
    "Signal": [
        st.Page("pages/1_Customer_Signal_Synthesizer.py", title="Customer Signal Synthesizer",
                icon=":material/forum:"),
        st.Page("pages/2_Behavioral_Friction_Analyzer.py", title="Behavioral Friction Analyzer",
                icon=":material/ads_click:"),
    ],
    "Diagnosis": [
        st.Page("pages/3_Qualified_Demand_Leakage_Auditor.py", title="Qualified Demand Leakage Auditor",
                icon=":material/filter_alt:"),
    ],
}

st.navigation(PAGES).run()
