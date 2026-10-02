"""Load the synthetic EchoAI datasets from data/synthetic/."""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

from gios import config

DATASETS = ("funnel_by_source", "web_behavior", "customer_evidence", "sales_feedback", "experiments")


def text_key(text: str) -> str:
    """Stable short id for a piece of text (used to dedupe and cache LLM tags)."""
    return "v" + hashlib.sha1(text.strip().encode()).hexdigest()[:10]


@lru_cache(maxsize=None)
def _read(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def load(name: str, data_dir: Optional[Path] = None) -> pd.DataFrame:
    if name not in DATASETS:
        raise KeyError(f"unknown dataset '{name}'")
    path = Path(data_dir or config.DATA_DIR) / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run `python scripts/generate_data.py`.")
    df = _read(str(path), path.stat().st_mtime).copy()
    if name == "customer_evidence":
        df.insert(0, "evidence_id", [f"E{i:04d}" for i in range(1, len(df) + 1)])
        df["text_key"] = df["verbatim"].map(text_key)
        df["month"] = df["date"].str[:7]
    if name == "web_behavior":
        df["month"] = df["date"].str[:7]
    if name == "sales_feedback":
        df["rejection_reason"] = df["rejection_reason"].fillna("")
    return df
