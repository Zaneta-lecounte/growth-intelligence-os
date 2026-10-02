"""Experiment Opportunity Scorer (prioritization layer).

Source of truth: specs/experiment-opportunity-scorer.md
"""
from gios.modules.experiment_opportunity_scorer.analysis import MODULE, Settings
from gios.modules.experiment_opportunity_scorer.pipeline import ScoredBacklog, build_backlog, save, score

__all__ = ["MODULE", "Settings", "ScoredBacklog", "build_backlog", "save", "score"]
