"""Behavioral Friction Analyzer (signal layer).

Source of truth: specs/behavioral-friction-analyzer.md
"""
from gios.modules.behavioral_friction_analyzer.analysis import MODULE, Thresholds
from gios.modules.behavioral_friction_analyzer.pipeline import BehavioralResult, run, to_signals

__all__ = ["MODULE", "Thresholds", "BehavioralResult", "run", "to_signals"]
