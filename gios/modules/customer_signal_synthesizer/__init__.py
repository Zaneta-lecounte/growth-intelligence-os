"""Customer Signal Synthesizer (signal layer).

Source of truth: specs/customer-signal-synthesizer.md
"""
from gios.modules.customer_signal_synthesizer.analysis import MODULE
from gios.modules.customer_signal_synthesizer.pipeline import CustomerSignalResult, run

__all__ = ["MODULE", "CustomerSignalResult", "run"]
