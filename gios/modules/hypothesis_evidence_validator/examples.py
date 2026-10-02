"""Prefilled example for manual entry: a plausible but assumption-only hypothesis."""
from gios.core.schemas import Hypothesis

ASSUMPTION_ONLY = Hypothesis(
    id="hev-example-chatbot",
    title="Homepage chatbot will lift demo requests",
    observed_problem="Homepage visitors do not request demos as often as we would like",
    affected_audience="All homepage visitors",
    causal_explanation="Visitors have unanswered questions and nobody to ask in the moment",
    intervention="Add an AI chatbot to the homepage",
    expected_behavior_change="Visitors chat with the bot and then request a demo",
    expected_business_outcome="More demo requests and more qualified pipeline",
)
