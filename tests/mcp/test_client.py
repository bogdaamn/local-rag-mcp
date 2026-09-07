import threading

from mcp.client import MCPClient
from telemetry import context, storage


def _make_bare_client(send_fn):
    instance = MCPClient.__new__(MCPClient)
    instance.lock = threading.Lock()
    instance.next_id = 1
    instance._send = send_fn
    return instance


def test_call_tool_records_tool_telemetry():
    context.start_task("task-abc")
    context.next_turn()

    def fake_send(payload):
        assert payload["method"] == "tools/call"
        assert payload["params"]["name"] == "list_documents"
        return {"jsonrpc": "2.0", "id": payload["id"], "result": "doc1\ndoc2\ndoc3"}

    client = _make_bare_client(fake_send)

    response = client.call_tool("list_documents", {})

    assert response["result"] == "doc1\ndoc2\ndoc3"

    conn = storage.get_connection()
    try:
        row = conn.execute(
            "SELECT tool_name, task_id, output_tokens FROM tool_calls"
        ).fetchone()
    finally:
        conn.close()

    assert row == ("list_documents", "task-abc", len("doc1\ndoc2\ndoc3") // 4)


def test_call_tool_increments_next_id():
    def fake_send(payload):
        return {"jsonrpc": "2.0", "id": payload["id"], "result": ""}

    client = _make_bare_client(fake_send)
    assert client.next_id == 1

    client.call_tool("list_documents", {})

    assert client.next_id == 2
