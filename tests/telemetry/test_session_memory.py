from telemetry import session_memory


def test_get_memory_returns_empty_shape_when_nothing_recorded(tmp_path):
    db_path = tmp_path / "telemetry.db"

    memory = session_memory.get_memory("t1", db_path=db_path)

    assert memory == {
        "session_id": "t1", "task_id": "t1", "status": "in_progress",
        "decisions": [], "changes": [], "errors": [],
        "artifacts": {"relevant_files": [], "references": []},
    }


def test_record_decision_appends_to_decisions_list(tmp_path):
    db_path = tmp_path / "telemetry.db"

    session_memory.record_decision("t1", "used heuristic", reason="no LLM needed", db_path=db_path)

    memory = session_memory.get_memory("t1", db_path=db_path)
    assert len(memory["decisions"]) == 1
    assert memory["decisions"][0]["description"] == "used heuristic"
    assert memory["decisions"][0]["reason"] == "no LLM needed"
    assert memory["decisions"][0]["timestamp"]


def test_record_change_appends_to_changes_list(tmp_path):
    db_path = tmp_path / "telemetry.db"

    session_memory.record_change("t1", "output truncated", db_path=db_path)

    memory = session_memory.get_memory("t1", db_path=db_path)
    assert len(memory["changes"]) == 1
    assert memory["changes"][0]["description"] == "output truncated"
    assert memory["changes"][0]["timestamp"]


def test_record_error_appends_to_errors_list(tmp_path):
    db_path = tmp_path / "telemetry.db"

    session_memory.record_error("t1", "tool call failed", resolution="retried", db_path=db_path)

    memory = session_memory.get_memory("t1", db_path=db_path)
    assert memory["errors"][0]["description"] == "tool call failed"
    assert memory["errors"][0]["resolution"] == "retried"


def test_multiple_records_accumulate_in_order(tmp_path):
    db_path = tmp_path / "telemetry.db"

    session_memory.record_decision("t1", "first", db_path=db_path)
    session_memory.record_decision("t1", "second", db_path=db_path)

    memory = session_memory.get_memory("t1", db_path=db_path)
    assert [d["description"] for d in memory["decisions"]] == ["first", "second"]


def test_different_task_ids_do_not_share_memory(tmp_path):
    db_path = tmp_path / "telemetry.db"

    session_memory.record_decision("t1", "for t1", db_path=db_path)
    session_memory.record_decision("t2", "for t2", db_path=db_path)

    assert [d["description"] for d in session_memory.get_memory("t1", db_path=db_path)["decisions"]] == ["for t1"]
    assert [d["description"] for d in session_memory.get_memory("t2", db_path=db_path)["decisions"]] == ["for t2"]


def test_record_decision_does_not_raise_when_storage_fails(tmp_path, monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr("telemetry.storage.upsert_session_memory", _boom)

    session_memory.record_decision("t1", "x", db_path=tmp_path / "telemetry.db")  # must not raise
