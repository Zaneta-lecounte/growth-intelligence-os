from gios.core.schemas import Signal
from gios.modules.signal_layer import run_signal_layer


def test_run_signal_layer_seeds_store(store):
    counts = run_signal_layer(store)
    assert counts == {"customer_signal_synthesizer": 10, "behavioral_friction_analyzer": 2,
                      "qualified_demand_leakage_auditor": 14}
    signals = {s.id: s for s in store.list(Signal)}
    assert {"css-pricing_uncertainty", "bfa-pricing-source-paid_search", "qdl-flag-rising_cac-paid_search"} <= set(signals)
    assert signals["bfa-pricing-source-paid_search"].channel == "paid_search"
    assert "supported by customer evidence" in signals["bfa-pricing-source-paid_search"].summary
    assert signals["bfa-demo-device-mobile"].attributes == {"page": "demo", "device": "mobile",
                                                            "friction_type": "technical", "cause_status": "supported"}
    assert signals["qdl-leak-webinar-mql-to-sql"].attributes == {"kind": "leak", "source": "webinar",
                                                                 "stage": "MQL→SQL", "owner": "qualification"}
    assert signals["qdl-flag-rising_cac-paid_search"].attributes["owner"] == "acquisition"
    assert signals["css-pricing_uncertainty"].attributes["theme"] == "pricing_uncertainty"
    assert signals["css-usability_polish"].attributes["frequency_rank"] == "1"
    run_signal_layer(store)  # idempotent
    assert len(store.list(Signal)) == 26
