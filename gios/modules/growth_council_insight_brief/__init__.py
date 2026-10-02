"""Growth Council Insight Brief (learning layer).

Source of truth: specs/growth-council-insight-brief.md
"""
from gios.modules.growth_council_insight_brief.analysis import MODULE, gather
from gios.modules.growth_council_insight_brief.pipeline import Brief, run, slack, write

__all__ = ["MODULE", "Brief", "gather", "run", "slack", "write"]
