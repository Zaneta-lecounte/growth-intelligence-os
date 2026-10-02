"""Run all Phase 1 (signal-layer) modules at default settings and save their Signals.

Used to seed the store for the diagnosis layer, from the UI or tests.
"""
from __future__ import annotations

from typing import Any, Optional

from gios.core.store import Store


def run_signal_layer(store: Optional[Store] = None, client: Any = None) -> dict[str, int]:
    from gios.modules import behavioral_friction_analyzer as bfa
    from gios.modules import customer_signal_synthesizer as css
    from gios.modules import qualified_demand_leakage_auditor as qdl
    from gios.modules.customer_signal_synthesizer import analysis as css_analysis

    store = store or Store()
    customer_result = css.run(client=client)
    customer = css_analysis.to_signals(customer_result.themes, period=customer_result.meta["period"])
    behavioral = bfa.to_signals(bfa.run(customer_signals=customer, client=client))
    leakage = qdl.to_signals(qdl.run(client=client))
    counts = {}
    for module, signals in ((css.MODULE, customer), (bfa.MODULE, behavioral), (qdl.MODULE, leakage)):
        store.replace_module_output(signals, module=module)
        counts[module] = len(signals)
    return counts
