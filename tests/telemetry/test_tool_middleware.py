import sqlite3

from telemetry import context
from telemetry.tool_middleware import record_tool_call


def test_records_a_tool_calls_row_and_returns_fn_result(tmp_path):
    db_path = tmp_path / "telemetry.db"
    context.start_task("t1")
    context.next_turn()

    response = record_tool_call(
        "list_documents", {}, lambda: {"result": "doc1\ndoc2"}, db_path=db_path,
    )

    assert response == {"result": "doc1\ndoc2"}

    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT task_id, turn_number, tool_name, output_tokens FROM tool_calls"
    ).fetchone()
    conn.close()
    assert row == ("t1", 1, "list_documents", len("doc1\ndoc2") // 4)


def test_input_size_is_byte_length_of_serialized_arguments(tmp_path):
    db_path = tmp_path / "telemetry.db"
    context.start_task("t2")
    context.next_turn()

    record_tool_call(
        "read_document", {"file_path": "docs/x.md"}, lambda: {"result": "y"}, db_path=db_path,
    )

    import json
    expected_size = len(json.dumps({"file_path": "docs/x.md"}).encode("utf-8"))

    conn = sqlite3.connect(str(db_path))
    input_size = conn.execute("SELECT input_size FROM tool_calls").fetchone()[0]
    conn.close()
    assert input_size == expected_size


def test_output_size_is_byte_length_of_the_result_field_only(tmp_path):
    db_path = tmp_path / "telemetry.db"
    context.start_task("t3")
    context.next_turn()

    record_tool_call(
        "search_documents", {"query": "vacation"},
        lambda: {"jsonrpc": "2.0", "id": 1, "result": "found: vacation-policy.md"},
        db_path=db_path,
    )

    conn = sqlite3.connect(str(db_path))
    output_size = conn.execute("SELECT output_size FROM tool_calls").fetchone()[0]
    conn.close()
    assert output_size == len("found: vacation-policy.md".encode("utf-8"))


def test_records_and_does_not_raise_when_no_ambient_task(tmp_path):
    db_path = tmp_path / "telemetry.db"
    context.current_task_id.set(None)

    response = record_tool_call(
        "list_documents", {}, lambda: {"result": "doc1\ndoc2"}, db_path=db_path,
    )

    assert response == {"result": "doc1\ndoc2"}

    conn = sqlite3.connect(str(db_path))
    task_id = conn.execute("SELECT task_id FROM tool_calls").fetchone()[0]
    conn.close()
    assert task_id == "unattributed"
