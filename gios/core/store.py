"""SQLite persistence so modules can pass outputs to each other.

Each entity type has its own table holding the validated model as JSON, plus the id,
the producing module, and a timestamp for filtering. Reads re-validate through Pydantic.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Optional, TypeVar, Union

from gios import config
from gios.core.schemas import (
    Experiment,
    GIOSModel,
    Hypothesis,
    Learning,
    Opportunity,
    Recommendation,
    Signal,
)

M = TypeVar("M", bound=GIOSModel)

TABLES: dict[type[GIOSModel], str] = {
    Signal: "signals",
    Opportunity: "opportunities",
    Hypothesis: "hypotheses",
    Experiment: "experiments",
    Learning: "learnings",
    Recommendation: "recommendations",
}


def _table(model_cls: type[GIOSModel]) -> str:
    try:
        return TABLES[model_cls]
    except KeyError:
        raise TypeError(f"{model_cls.__name__} is not a stored GIOS entity") from None


class Store:
    def __init__(self, path: Union[str, Path, None] = None):
        self.path = str(path if path is not None else config.DB_PATH)
        with self._connect() as conn:
            for table in TABLES.values():
                conn.execute(
                    f"""CREATE TABLE IF NOT EXISTS {table} (
                        id TEXT PRIMARY KEY,
                        module TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL,
                        payload TEXT NOT NULL
                    )"""
                )
                conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_module ON {table}(module)")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def save(self, obj: GIOSModel, module: str = "") -> str:
        """Insert or replace an entity; returns its id."""
        return self.save_many([obj], module=module)[0]

    def save_many(self, objs: Iterable[GIOSModel], module: str = "") -> list[str]:
        ids = []
        with self._connect() as conn:
            for obj in objs:
                conn.execute(
                    f"INSERT OR REPLACE INTO {_table(type(obj))} (id, module, created_at, payload) "
                    "VALUES (?, ?, ?, ?)",
                    (obj.id, module, obj.created_at.isoformat(), obj.model_dump_json()),
                )
                ids.append(obj.id)
        return ids

    def get(self, model_cls: type[M], obj_id: str) -> Optional[M]:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT payload FROM {_table(model_cls)} WHERE id = ?", (obj_id,)
            ).fetchone()
        return model_cls.model_validate_json(row[0]) if row else None

    def list(self, model_cls: type[M], module: Optional[str] = None) -> list[M]:
        sql = f"SELECT payload FROM {_table(model_cls)}"
        params: tuple = ()
        if module is not None:
            sql += " WHERE module = ?"
            params = (module,)
        sql += " ORDER BY created_at, rowid"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [model_cls.model_validate_json(r[0]) for r in rows]

    def delete(self, model_cls: type[GIOSModel], obj_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(f"DELETE FROM {_table(model_cls)} WHERE id = ?", (obj_id,))
        return cur.rowcount > 0

    def clear(self, model_cls: Optional[type[GIOSModel]] = None) -> None:
        tables = [_table(model_cls)] if model_cls else list(TABLES.values())
        with self._connect() as conn:
            for table in tables:
                conn.execute(f"DELETE FROM {table}")

    def counts(self) -> dict[str, int]:
        with self._connect() as conn:
            return {
                table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in TABLES.values()
            }
