import pytest

from telemetry import context


@pytest.fixture(autouse=True)
def isolated_telemetry_db(tmp_path, monkeypatch):
    """Redirect every test's telemetry writes to a per-test tmp file so test
    runs never touch (or get polluted by) the real project telemetry.db.
    storage.py re-imports config.TELEMETRY_DB_PATH locally on every call, so
    this monkeypatch takes effect even for code that doesn't accept a
    db_path override.

    Also starts a fresh ambient task via context.start_task() so any test
    that (directly or transitively, e.g. through ask_llm() ->
    record_llm_call()) inserts an llm_calls/tool_calls row gets a non-null
    task_id without needing to call context.start_task() itself. Tests that
    care about a specific task_id/turn_number still call context.start_task()
    explicitly, which simply overrides this default."""
    monkeypatch.setattr("config.TELEMETRY_DB_PATH", str(tmp_path / "telemetry.db"))
    context.start_task()
