"""Downstream Impact Analyzer pipeline: load an experiment (CSV or manual) -> analyze (code) ->
save the result on an Experiment record."""
from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd

from gios.core import data
from gios.core.schemas import Experiment, Variant
from gios.core.store import Store
from gios.modules.downstream_impact_analyzer import analysis
from gios.modules.downstream_impact_analyzer.analysis import ImpactResult, Settings

VALUE_COLUMNS = analysis.COUNTS + ["revenue", "spend"]


def experiment_ids() -> list[str]:
    return list(dict.fromkeys(data.load("experiments").experiment_id))


def load(experiment_id: str) -> dict:
    rows = data.load("experiments")
    rows = rows[rows.experiment_id == experiment_id]
    if rows.empty:
        raise KeyError(f"unknown experiment {experiment_id}")
    first = rows.iloc[0]
    cells = data.load("experiment_segments")
    cells = cells[cells.experiment_id == experiment_id]
    return {
        "experiment_id": experiment_id, "name": first.experiment_name, "page": first.page,
        "hypothesis": first.hypothesis, "primary_metric": first.primary_metric,
        "start": date.fromisoformat(first.start_date), "end": date.fromisoformat(first.end_date),
        "observed_through": date.fromisoformat(first.observed_through),
        "variants": rows.set_index("variant")[VALUE_COLUMNS], "cells": cells,
    }


def run(experiment_id: str, settings: Settings = Settings()) -> ImpactResult:
    e = load(experiment_id)
    velocity = analysis.velocity_for_mix(e["cells"], data.load("lead_velocity"))
    return analysis.analyze(experiment_id, e["name"], e["variants"], e["cells"], velocity, e["start"],
                            e["observed_through"], settings,
                            meta={k: e[k] for k in ("page", "hypothesis", "primary_metric", "start", "end",
                                                    "observed_through")})


def run_manual(name: str, variants: pd.DataFrame, start: Optional[date], observed_through: Optional[date],
               velocity_days: Optional[float], settings: Settings = Settings(), page: str = "",
               hypothesis: str = "") -> ImpactResult:
    eid = "manual-" + "".join(ch for ch in name.lower() if ch.isalnum())[:24]
    return analysis.analyze(eid, name, variants[VALUE_COLUMNS], None, velocity_days, start, observed_through,
                            settings, meta={"page": page, "hypothesis": hypothesis, "primary_metric": "conversion",
                                            "start": start, "end": None, "observed_through": observed_through})


def to_experiment(result: ImpactResult, variants: pd.DataFrame) -> Experiment:
    from gios.modules.downstream_impact_analyzer.report import statistical_summary

    m = result.meta
    return Experiment(
        id=result.experiment_id, name=result.name, hypothesis_text=m.get("hypothesis", ""),
        primary_metric=m.get("primary_metric", "conversion"), page=m.get("page", ""),
        start_date=m.get("start"), end_date=m.get("end"), observed_through=m.get("observed_through"),
        status="completed",
        variants=[Variant(name=str(v), **{k: (float(r[k]) if k in ("revenue", "spend") else int(r[k]))
                                          for k in VALUE_COLUMNS}) for v, r in variants.iterrows()],
        interpretation=result.interpretation, impact_recommendation=result.recommendation,
        statistical_result=statistical_summary(result),
        segment_findings=result.segment_notes or [f.describe() for f in result.segments[:3]],
    )


def save(result: ImpactResult, variants: pd.DataFrame, store: Store) -> str:
    return store.save(to_experiment(result, variants), module=analysis.MODULE)


def analyze_all(store: Store, settings: Settings = Settings()) -> list[str]:
    ids = []
    for eid in experiment_ids():
        ids.append(save(run(eid, settings), load(eid)["variants"], store))
    return ids
