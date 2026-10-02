"""Qualified Demand Leakage Auditor (diagnosis layer; runs in the signal phase).

Source of truth: specs/qualified-demand-leakage-auditor.md
"""
from gios.modules.qualified_demand_leakage_auditor.analysis import MODULE, Settings
from gios.modules.qualified_demand_leakage_auditor.pipeline import LeakageResult, run, to_signals

__all__ = ["MODULE", "Settings", "LeakageResult", "run", "to_signals"]
