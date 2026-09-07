import rag.query as query


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def test_ask_llm_without_temperature_or_timeout_preserves_existing_behavior(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse({"response": "answer"})

    monkeypatch.setattr(query.requests, "post", fake_post)

    result = query.ask_llm("hello")

    assert result == "answer"
    assert "options" not in captured["json"]
    assert captured["timeout"] is None


def test_ask_llm_with_temperature_adds_options_payload(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["json"] = json
        return _FakeResponse({"response": "answer"})

    monkeypatch.setattr(query.requests, "post", fake_post)

    query.ask_llm("hello", temperature=0.1)

    assert captured["json"]["options"] == {"temperature": 0.1}


def test_ask_llm_with_timeout_passes_it_through(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["timeout"] = timeout
        return _FakeResponse({"response": "answer"})

    monkeypatch.setattr(query.requests, "post", fake_post)

    query.ask_llm("hello", timeout=8.0)

    assert captured["timeout"] == 8.0


# config.TELEMETRY_DB_PATH is an absolute tmp path once tests/conftest.py's
# isolated_telemetry_db autouse fixture has run for this test, so
# sqlite3.connect(config.TELEMETRY_DB_PATH) opens the exact file ask_llm()'s
# telemetry write used.
import config
import sqlite3


def test_ask_llm_default_call_site_is_ask_llm(monkeypatch):
    monkeypatch.setattr(query.requests, "post", lambda *a, **k: _FakeResponse({"response": "answer"}))

    query.ask_llm("hello")

    conn = sqlite3.connect(config.TELEMETRY_DB_PATH)
    row = conn.execute("SELECT call_site FROM llm_calls ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    assert row == ("ask_llm",)


def test_ask_llm_extracts_token_counts_from_ollama_response(monkeypatch):
    monkeypatch.setattr(
        query.requests, "post",
        lambda *a, **k: _FakeResponse({"response": "answer", "prompt_eval_count": 11, "eval_count": 22}),
    )

    query.ask_llm("hello")

    conn = sqlite3.connect(config.TELEMETRY_DB_PATH)
    row = conn.execute(
        "SELECT input_tokens, output_tokens FROM llm_calls ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    assert row == (11, 22)


def test_ask_llm_accepts_an_explicit_call_site(monkeypatch):
    monkeypatch.setattr(query.requests, "post", lambda *a, **k: _FakeResponse({"response": "answer"}))

    query.ask_llm("hello", call_site="query_expansion")

    conn = sqlite3.connect(config.TELEMETRY_DB_PATH)
    row = conn.execute("SELECT call_site FROM llm_calls ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    assert row == ("query_expansion",)
