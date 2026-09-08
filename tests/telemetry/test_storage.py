import sqlite3

from telemetry import storage


def test_get_connection_creates_all_three_tables(tmp_path):
    db_path = tmp_path / "telemetry.db"

    conn = storage.get_connection(db_path)
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    conn.close()

    assert {"llm_calls", "tool_calls", "llm_cache"} <= tables


def test_get_connection_is_idempotent_across_calls(tmp_path):
    db_path = tmp_path / "telemetry.db"

    storage.get_connection(db_path).close()
    storage.get_connection(db_path).close()  # must not raise on re-create


def test_get_connection_with_no_db_path_resolves_from_config(tmp_path, monkeypatch):
    monkeypatch.setattr("config.TELEMETRY_DB_PATH", str(tmp_path / "default.db"))

    conn = storage.get_connection()
    conn.close()

    assert (tmp_path / "default.db").exists()


def test_insert_llm_call_writes_a_row(tmp_path):
    db_path = tmp_path / "telemetry.db"

    storage.insert_llm_call(
        agent_id="company-kb-assistant", task_id="t1", turn_number=1,
        model="qwen3:0.6b", input_tokens=10, output_tokens=20,
        cached_tokens=0, reasoning_tokens=0, latency_ms=123.4,
        estimated_cost=0.000006, call_site="ask_llm", db_path=db_path,
    )

    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT agent_id, task_id, turn_number, model, input_tokens, "
        "output_tokens, cached_tokens, reasoning_tokens, call_site "
        "FROM llm_calls"
    ).fetchone()
    conn.close()

    assert row == (
        "company-kb-assistant", "t1", 1, "qwen3:0.6b", 10, 20, 0, 0, "ask_llm",
    )


def test_insert_llm_call_defaults_timestamp_to_now_iso(tmp_path):
    db_path = tmp_path / "telemetry.db"

    storage.insert_llm_call(
        agent_id="a", task_id="t1", turn_number=1, model="m",
        input_tokens=1, output_tokens=1, cached_tokens=0, reasoning_tokens=0,
        latency_ms=1.0, estimated_cost=0.0, call_site="ask_llm", db_path=db_path,
    )

    conn = sqlite3.connect(str(db_path))
    timestamp = conn.execute("SELECT timestamp FROM llm_calls").fetchone()[0]
    conn.close()

    assert timestamp  # non-empty ISO string


def test_insert_tool_call_writes_a_row(tmp_path):
    db_path = tmp_path / "telemetry.db"

    storage.insert_tool_call(
        agent_id="company-kb-assistant", task_id="t1", turn_number=2,
        tool_name="list_documents", input_size=2, output_size=50,
        output_tokens=12, duration_ms=45.0, db_path=db_path,
    )

    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT agent_id, task_id, turn_number, tool_name, input_size, "
        "output_size, output_tokens FROM tool_calls"
    ).fetchone()
    conn.close()

    assert row == ("company-kb-assistant", "t1", 2, "list_documents", 2, 50, 12)


def test_get_connection_creates_session_memory_table(tmp_path):
    db_path = tmp_path / "telemetry.db"

    conn = storage.get_connection(db_path)
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    conn.close()

    assert "session_memory" in tables


def test_get_connection_adds_truncated_column_to_a_pre_existing_tool_calls_table(tmp_path):
    db_path = tmp_path / "telemetry.db"
    # Simulate a DB created before this migration: tool_calls without `truncated`.
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE tool_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            turn_number INTEGER NOT NULL,
            tool_name TEXT NOT NULL,
            input_size INTEGER NOT NULL,
            output_size INTEGER NOT NULL,
            output_tokens INTEGER NOT NULL,
            duration_ms REAL NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()

    conn = storage.get_connection(db_path)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(tool_calls)")}
    conn.close()

    assert "truncated" in columns


def test_insert_tool_call_defaults_truncated_to_false(tmp_path):
    db_path = tmp_path / "telemetry.db"

    storage.insert_tool_call(
        agent_id="a", task_id="t1", turn_number=1, tool_name="read_document",
        input_size=1, output_size=1, output_tokens=1, duration_ms=1.0, db_path=db_path,
    )

    conn = sqlite3.connect(str(db_path))
    truncated = conn.execute("SELECT truncated FROM tool_calls").fetchone()[0]
    conn.close()
    assert truncated == 0


def test_insert_tool_call_stores_truncated_true(tmp_path):
    db_path = tmp_path / "telemetry.db"

    storage.insert_tool_call(
        agent_id="a", task_id="t1", turn_number=1, tool_name="read_document",
        input_size=1, output_size=1, output_tokens=1, duration_ms=1.0,
        truncated=True, db_path=db_path,
    )

    conn = sqlite3.connect(str(db_path))
    truncated = conn.execute("SELECT truncated FROM tool_calls").fetchone()[0]
    conn.close()
    assert truncated == 1


def test_upsert_session_memory_then_get_roundtrips(tmp_path):
    db_path = tmp_path / "telemetry.db"

    storage.upsert_session_memory("t1", '{"a": 1}', db_path=db_path)

    assert storage.get_session_memory("t1", db_path=db_path) == '{"a": 1}'


def test_upsert_session_memory_overwrites_existing_row(tmp_path):
    db_path = tmp_path / "telemetry.db"

    storage.upsert_session_memory("t1", '{"a": 1}', db_path=db_path)
    storage.upsert_session_memory("t1", '{"a": 2}', db_path=db_path)

    assert storage.get_session_memory("t1", db_path=db_path) == '{"a": 2}'


def test_get_session_memory_returns_none_when_absent(tmp_path):
    db_path = tmp_path / "telemetry.db"
    assert storage.get_session_memory("missing", db_path=db_path) is None
