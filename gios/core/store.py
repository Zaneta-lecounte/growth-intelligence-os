"""SQLite persistence so modules can pass outputs to each other.

Each entity type has its own table holding the validated model as JSON, plus the id,
the producing module, and a timestamp for filtering. Reads re-validate through Pydantic.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
import logging
from typing import Iterable, Iterator, Optional, TypeVar, Union

from pydantic import ValidationError

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
log = logging.getLogger(__name__)

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

    def save(self, obj: GIOSModel, module: Optional[str] = None) -> str:
        """Insert or replace an entity; returns its id. module=None keeps the module an
        existing record was saved under ("" for new records)."""
        return self.save_many([obj], module=module)[0]

    def save_many(self, objs: Iterable[GIOSModel], module: Optional[str] = None) -> list[str]:
        ids = []
        with self._connect() as conn:
            for obj in objs:
                mod = module
                if mod is None:
                    row = conn.execute(f"SELECT module FROM {_table(type(obj))} WHERE id = ?",
                                       (obj.id,)).fetchone()
                    mod = row[0] if row else ""
                conn.execute(
                    f"INSERT OR REPLACE INTO {_table(type(obj))} (id, module, created_at, payload) "
                    "VALUES (?, ?, ?, ?)",
                    (obj.id, mod, obj.created_at.isoformat(), obj.model_dump_json()),
                )
                ids.append(obj.id)
        return ids

    @staticmethod
    def _parse(model_cls: type[M], payload: str) -> Optional[M]:
        """Validate a stored payload; rows written under an incompatible schema are skipped
        (and logged) rather than taking every page down."""
        try:
            return model_cls.model_validate_json(payload)
        except ValidationError as exc:
            log.warning("skipping unreadable %s record: %s", model_cls.__name__, exc.errors()[0]["msg"])
            return None

    def get(self, model_cls: type[M], obj_id: str) -> Optional[M]:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT payload FROM {_table(model_cls)} WHERE id = ?", (obj_id,)
            ).fetchone()
        return self._parse(model_cls, row[0]) if row else None

    def list(self, model_cls: type[M], module: Optional[str] = None) -> list[M]:
        sql = f"SELECT payload FROM {_table(model_cls)}"
        params: tuple = ()
        if module is not None:
            sql += " WHERE module = ?"
            params = (module,)
        sql += " ORDER BY created_at, rowid"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        parsed = (self._parse(model_cls, r[0]) for r in rows)
        return [p for p in parsed if p is not None]

    def unreadable(self) -> dict[str, int]:
        """Rows per table that no longer validate against the current schema."""
        out = {}
        with self._connect() as conn:
            for model_cls, table in TABLES.items():
                bad = 0
                for (payload,) in conn.execute(f"SELECT payload FROM {table}"):
                    try:
                        model_cls.model_validate_json(payload)
                    except ValidationError:
                        bad += 1
                if bad:
                    out[table] = bad
        return out

    def delete(self, model_cls: type[GIOSModel], obj_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(f"DELETE FROM {_table(model_cls)} WHERE id = ?", (obj_id,))
        return cur.rowcount > 0

    def delete_by_module(self, model_cls: type[GIOSModel], module: str) -> int:
        with self._connect() as conn:
            cur = conn.execute(f"DELETE FROM {_table(model_cls)} WHERE module = ?", (module,))
        return cur.rowcount

    def module_of(self, model_cls: type[GIOSModel], obj_id: str) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(f"SELECT module FROM {_table(model_cls)} WHERE id = ?", (obj_id,)).fetchone()
        return row[0] if row else None

    def replace_module_output(self, objs: Iterable[GIOSModel], module: str) -> list[str]:
        """Replace everything `module` previously saved for these entity types."""
        objs = list(objs)
        for model_cls in {type(o) for o in objs}:
            self.delete_by_module(model_cls, module)
        return self.save_many(objs, module=module)

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
