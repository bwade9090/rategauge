"""SQLite tracing for LLM calls made through the service.

One row per extraction request: what was asked, plus the numbers that matter
for monitoring — token usage, computed USD cost, latency, and outcome. The
database is runtime state (gitignored) and is created on first use; every
operation opens its own connection, so concurrent request threads are safe.
"""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

TRACES_PATH = Path("eval/traces.sqlite")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_utc TEXT NOT NULL,
    doc_id TEXT,
    bank TEXT NOT NULL,
    model_key TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    latency_ms INTEGER NOT NULL,
    ok INTEGER NOT NULL,
    error TEXT
)
"""


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute(_SCHEMA)
    return connection


def record_trace(
    path: Path,
    *,
    doc_id: str | None,
    bank: str,
    model_key: str,
    prompt_version: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    latency_ms: int,
    ok: bool,
    error: str | None,
) -> int:
    """Persist one trace row; returns its id (surfaced to the API caller)."""
    connection = _connect(path)
    try:
        with connection:  # one transaction
            cursor = connection.execute(
                "INSERT INTO traces (timestamp_utc, doc_id, bank, model_key, prompt_version,"
                " input_tokens, output_tokens, cost_usd, latency_ms, ok, error)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    datetime.now(UTC).isoformat(timespec="seconds"),
                    doc_id,
                    bank,
                    model_key,
                    prompt_version,
                    input_tokens,
                    output_tokens,
                    round(cost_usd, 6),
                    latency_ms,
                    int(ok),
                    error,
                ),
            )
            return int(cursor.lastrowid)
    finally:
        connection.close()


def recent_traces(path: Path, *, limit: int = 50) -> list[dict]:
    """Most recent trace rows, newest first."""
    if not path.exists():
        return []
    connection = _connect(path)
    try:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM traces ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()
