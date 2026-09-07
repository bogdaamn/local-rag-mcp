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
