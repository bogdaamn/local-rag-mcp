import json

import assistant as assistant_module
from assistant import CompanyKBAssistant
from telemetry import context, session_memory


class _FakeMCPClient:
    def __init__(self, *args, **kwargs):
        pass

    def close(self):
        pass


def _make_bare_assistant(mcp=None, task_id="test-task"):
    instance = CompanyKBAssistant.__new__(CompanyKBAssistant)
    instance.mcp = mcp
    instance.task_id = task_id
    return instance


def test_init_starts_a_task_in_ambient_context(monkeypatch):
    monkeypatch.setattr(assistant_module, "MCPClient", _FakeMCPClient)

    instance = CompanyKBAssistant()

    assert instance.task_id == context.current_task_id.get()
    assert context.current_turn_number.get() == 0


def test_query_advances_turn_number_once_per_call(monkeypatch):
    monkeypatch.setattr(assistant_module, "MCPClient", _FakeMCPClient)
    instance = CompanyKBAssistant()

    monkeypatch.setattr(assistant_module, "retrieve", lambda q: [])
    monkeypatch.setattr(assistant_module, "build_prompt", lambda q, c: "prompt")
    monkeypatch.setattr(assistant_module, "ask_llm", lambda prompt: "answer")
    monkeypatch.setattr(assistant_module, "decide_tool_usage", lambda q, c: (None, None))

    instance.query("first question")
    assert context.current_turn_number.get() == 1

    instance.query("second question")
    assert context.current_turn_number.get() == 2


def test_query_delegates_tool_decision_to_the_heuristic_not_an_llm_call(monkeypatch):
    monkeypatch.setattr(assistant_module, "MCPClient", _FakeMCPClient)
    instance = CompanyKBAssistant()

    captured = {}
    monkeypatch.setattr(assistant_module, "retrieve", lambda q: [])
    monkeypatch.setattr(assistant_module, "build_prompt", lambda q, c: "prompt")
    monkeypatch.setattr(assistant_module, "ask_llm", lambda prompt: "answer")

    def fake_decide(query, contexts):
        captured["query"] = query
        captured["contexts"] = contexts
        return None, None

    monkeypatch.setattr(assistant_module, "decide_tool_usage", fake_decide)

    instance.query("list all documents")

    assert captured["query"] == "list all documents"
    assert captured["contexts"] == []


def test_query_records_a_decision_in_session_memory(monkeypatch):
    monkeypatch.setattr(assistant_module, "MCPClient", _FakeMCPClient)
    instance = CompanyKBAssistant()

    monkeypatch.setattr(assistant_module, "retrieve", lambda q: [])
    monkeypatch.setattr(assistant_module, "build_prompt", lambda q, c: "prompt")
    monkeypatch.setattr(assistant_module, "ask_llm", lambda prompt: "answer")
    monkeypatch.setattr(assistant_module, "decide_tool_usage", lambda q, c: (None, None))

    instance.query("a pure RAG question")

    memory = session_memory.get_memory(instance.task_id)
    assert len(memory["decisions"]) == 1
    assert "no tool" in memory["decisions"][0]["description"].lower()


def test_extract_mcp_text_returns_plain_string_unchanged():
    instance = _make_bare_assistant()
    assert instance._extract_mcp_text("doc1\ndoc2") == "doc1\ndoc2"


def test_extract_mcp_text_unwraps_json_content_envelope():
    instance = _make_bare_assistant()
    payload = json.dumps({
        "content": "the file text", "truncated": False,
        "total_chars": 13, "next_offset": None,
    })
    assert instance._extract_mcp_text(payload) == "the file text"


def test_extract_mcp_text_appends_truncation_note_when_truncated():
    instance = _make_bare_assistant()
    payload = json.dumps({
        "content": "partial text", "truncated": True,
        "total_chars": 9000, "next_offset": 4000,
    })
    result = instance._extract_mcp_text(payload)
    assert result.startswith("partial text")
    assert "truncated" in result.lower()


def test_call_mcp_tool_records_an_error_in_session_memory_on_exception():
    class _BoomMCP:
        def call_tool(self, tool_name, tool_args):
            raise ConnectionError("mcp server gone")

    instance = _make_bare_assistant(mcp=_BoomMCP())

    instance._call_mcp_tool("read_document", {"file_path": "docs/x.txt"})

    memory = session_memory.get_memory(instance.task_id)
    assert len(memory["errors"]) == 1
    assert "read_document" in memory["errors"][0]["description"]


def test_query_records_a_change_when_tool_output_is_truncated(monkeypatch):
    monkeypatch.setattr(assistant_module, "MCPClient", _FakeMCPClient)
    instance = CompanyKBAssistant()

    monkeypatch.setattr(assistant_module, "retrieve", lambda q: [])
    monkeypatch.setattr(assistant_module, "build_prompt", lambda q, c: "prompt")
    monkeypatch.setattr(assistant_module, "ask_llm", lambda prompt: "answer")
    monkeypatch.setattr(
        assistant_module, "decide_tool_usage",
        lambda q, c: ("read_document", {"file_path": "docs/sqlite.txt"}),
    )
    monkeypatch.setattr(
        instance, "_call_mcp_tool",
        lambda tool_name, tool_args: json.dumps({
            "content": "partial", "truncated": True,
            "total_chars": 9000, "next_offset": 4000,
        }),
    )

    instance.query("read the full contents of sqlite.txt")

    memory = session_memory.get_memory(instance.task_id)
    assert len(memory["changes"]) == 1
    assert "truncated" in memory["changes"][0]["description"].lower()
