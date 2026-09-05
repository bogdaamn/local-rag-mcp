import assistant as assistant_module
from assistant import CompanyKBAssistant
from telemetry import context


class _FakeMCPClient:
    def __init__(self, *args, **kwargs):
        pass

    def close(self):
        pass


def _make_bare_assistant(llm_client=None, mcp=None):
    instance = CompanyKBAssistant.__new__(CompanyKBAssistant)
    instance.llm_client = llm_client
    instance.mcp = mcp
    return instance


def test_llm_decide_mcp_usage_returns_none_when_no_mcp_client():
    instance = _make_bare_assistant(llm_client=None, mcp=None)
    assert instance._llm_decide_mcp_usage("q", []) == (None, None)


def test_llm_decide_mcp_usage_wraps_chat_through_record_llm_call(monkeypatch):
    captured = {}

    def fake_record_llm_call(call_site, model, prompt, temperature, fn):
        captured["call_site"] = call_site
        captured["model"] = model
        captured["prompt"] = prompt
        captured["temperature"] = temperature
        text, input_tokens, output_tokens = fn()
        captured["fn_result"] = (text, input_tokens, output_tokens)
        return text

    monkeypatch.setattr(assistant_module, "record_llm_call", fake_record_llm_call)

    class _FakeLLMClient:
        def chat(self, model, messages):
            captured["chat_model"] = model
            captured["chat_messages"] = messages
            return {
                "message": {"content": '{"use_mcp": true, "tool": "list_documents", "args": {}}'},
                "prompt_eval_count": 12,
                "eval_count": 5,
            }

    instance = _make_bare_assistant(llm_client=_FakeLLMClient(), mcp=object())

    tool_name, tool_args = instance._llm_decide_mcp_usage("what docs exist?", [])

    assert tool_name == "list_documents"
    assert tool_args == {}
    assert captured["call_site"] == "mcp_decision"
    assert "what docs exist?" in captured["prompt"]
    assert captured["temperature"] is None
    assert captured["fn_result"] == (
        '{"use_mcp": true, "tool": "list_documents", "args": {}}', 12, 5,
    )


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
    monkeypatch.setattr(instance, "_llm_decide_mcp_usage", lambda q, c: (None, None))

    instance.query("first question")
    assert context.current_turn_number.get() == 1

    instance.query("second question")
    assert context.current_turn_number.get() == 2
