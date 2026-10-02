"""GIOS entry point: multipage navigation."""
import streamlit as st

PAGES = {
    "Overview": [st.Page("home.py", title="Home", icon=":material/home:", default=True)],
    "Signal": [
        st.Page("views/1_Customer_Signal_Synthesizer.py", title="Customer Signal Synthesizer",
                icon=":material/forum:"),
        st.Page("views/2_Behavioral_Friction_Analyzer.py", title="Behavioral Friction Analyzer",
                icon=":material/ads_click:"),
    ],
    "Diagnosis": [
        st.Page("views/3_Qualified_Demand_Leakage_Auditor.py", title="Qualified Demand Leakage Auditor",
                icon=":material/filter_alt:"),
        st.Page("views/4_Growth_Intelligence_Diagnostic.py", title="Growth Intelligence Diagnostic",
                icon=":material/troubleshoot:"),
        st.Page("views/5_Hypothesis_Evidence_Validator.py", title="Hypothesis Evidence Validator",
                icon=":material/fact_check:"),
    ],
    "Prioritization": [
        st.Page("views/6_Experiment_Opportunity_Scorer.py", title="Experiment Opportunity Scorer",
                icon=":material/leaderboard:"),
        st.Page("views/7_Growth_Priority_Orchestrator.py", title="Growth Priority Orchestrator",
                icon=":material/view_kanban:"),
    ],
    "Learning": [
        st.Page("views/8_Downstream_Impact_Analyzer.py", title="Downstream Impact Analyzer",
                icon=":material/query_stats:"),
    ],
}

st.navigation(PAGES).run()
