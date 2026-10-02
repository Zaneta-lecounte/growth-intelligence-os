"""Downstream Impact Analyzer (learning layer). Fully deterministic.

Source of truth: specs/downstream-impact-analyzer.md
"""
from gios.modules.downstream_impact_analyzer.analysis import MODULE, ImpactResult, Settings
from gios.modules.downstream_impact_analyzer.pipeline import analyze_all, experiment_ids, load, run, save

__all__ = ["MODULE", "ImpactResult", "Settings", "analyze_all", "experiment_ids", "load", "run", "save"]
