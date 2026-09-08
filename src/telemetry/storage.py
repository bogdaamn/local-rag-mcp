import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cached_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
    latency_ms REAL NOT NULL,
    estimated_cost REAL NOT NULL,
    call_site TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    tool_name TEXT NOT NULL,
    input_size INTEGER NOT NULL,
    output_size INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    duration_ms REAL NOT NULL,
    truncated INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    response TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS session_memory (
    task_id TEXT PRIMARY KEY,
    memory_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def _default_db_path():
    # Local import (not a module-level `from config import ...`) so tests
    # can monkeypatch config.TELEMETRY_DB_PATH and have it take effect here —
    # mirrors rag/build_index.py's FTS_DB_PATH pattern.
    from config import TELEMETRY_DB_PATH
    src_dir = Path(__file__).parent.parent
    return src_dir / TELEMETRY_DB_PATH


def _ensure_tool_calls_truncated_column(conn):
    """Migration for DBs created before the `truncated` column existed —
    CREATE TABLE IF NOT EXISTS is a no-op on an already-existing table, so
    new columns need an explicit ALTER TABLE guarded by a PRAGMA check
    (SQLite has no ADD COLUMN IF NOT EXISTS)."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(tool_calls)")}
    if "truncated" not in columns:
        conn.execute("ALTER TABLE tool_calls ADD COLUMN truncated INTEGER NOT NULL DEFAULT 0")
        conn.commit()


def get_connection(db_path=None):
    """Open a new connection to the telemetry DB, creating the schema (and
    migrating older schemas) if needed. Callers are responsible for closing
    it."""
    resolved = Path(db_path) if db_path else _default_db_path()
    conn = sqlite3.connect(str(resolved))
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    _ensure_tool_calls_truncated_column(conn)
    return conn


def insert_llm_call(
    agent_id, task_id, turn_number, model, input_tokens, output_tokens,
    cached_tokens, reasoning_tokens, latency_ms, estimated_cost, call_site,
    timestamp=None, db_path=None,
):
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            INSERT INTO llm_calls (
                timestamp, agent_id, task_id, turn_number, model,
                input_tokens, output_tokens, cached_tokens, reasoning_tokens,
                latency_ms, estimated_cost, call_site
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp or now_iso(), agent_id, task_id, turn_number, model,
                input_tokens, output_tokens, cached_tokens, reasoning_tokens,
                latency_ms, estimated_cost, call_site,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def insert_tool_call(
    agent_id, task_id, turn_number, tool_name, input_size, output_size,
    output_tokens, duration_ms, truncated=False, timestamp=None, db_path=None,
):
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            INSERT INTO tool_calls (
                timestamp, agent_id, task_id, turn_number, tool_name,
                input_size, output_size, output_tokens, duration_ms, truncated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp or now_iso(), agent_id, task_id, turn_number, tool_name,
                input_size, output_size, output_tokens, duration_ms,
                int(bool(truncated)),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_session_memory(task_id, db_path=None):
    """The stored memory_json string for task_id, or None if nothing has
    been recorded yet."""
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT memory_json FROM session_memory WHERE task_id = ?", (task_id,)
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def upsert_session_memory(task_id, memory_json, timestamp=None, db_path=None):
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            INSERT INTO session_memory (task_id, memory_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                memory_json = excluded.memory_json,
                updated_at = excluded.updated_at
            """,
            (task_id, memory_json, timestamp or now_iso()),
        )
        conn.commit()
    finally:
        conn.close()
