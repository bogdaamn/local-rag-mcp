from telemetry import storage
from telemetry.turn_summary import compute_turn_summary, format_turn_summary


def _llm(db_path, task_id, turn_number, input_tokens, output_tokens, cost, latency_ms, cached_tokens=0, call_site="ask_llm"):
    storage.insert_llm_call(
        agent_id="a", task_id=task_id, turn_number=turn_number, model="qwen3:0.6b",
        input_tokens=input_tokens, output_tokens=output_tokens, cached_tokens=cached_tokens,
        reasoning_tokens=0, latency_ms=latency_ms, estimated_cost=cost, call_site=call_site,
        db_path=db_path,
    )


def _tool(db_path, task_id, turn_number, duration_ms):
    storage.insert_tool_call(
        agent_id="a", task_id=task_id, turn_number=turn_number, tool_name="list_documents",
        input_size=2, output_size=10, output_tokens=3, duration_ms=duration_ms, db_path=db_path,
    )


def test_compute_turn_summary_aggregates_multiple_llm_calls_in_the_turn(tmp_path):
    db_path = tmp_path / "telemetry.db"
    _llm(db_path, "t1", 1, 40, 10, 0.001, 500.0, call_site="query_expansion")
    _llm(db_path, "t1", 1, 600, 300, 0.002, 1500.0, call_site="mcp_decision")
    _llm(db_path, "t1", 1, 3000, 200, 0.003, 2000.0, call_site="ask_llm")

    summary = compute_turn_summary(db_path, "t1", 1)

    assert summary["input_tokens"] == 3640
    assert summary["output_tokens"] == 510
    assert summary["total_tokens"] == 4150
    assert summary["estimated_cost"] == 0.006
    assert summary["latency_ms"] == 4000.0
    assert summary["llm_calls"] == 3
    assert summary["cache_hit"] is False


def test_compute_turn_summary_excludes_other_turns_and_tasks(tmp_path):
    db_path = tmp_path / "telemetry.db"
    _llm(db_path, "t1", 1, 100, 100, 0.01, 100.0)
    _llm(db_path, "t1", 2, 999, 999, 9.99, 999.0)
    _llm(db_path, "t2", 1, 999, 999, 9.99, 999.0)

    summary = compute_turn_summary(db_path, "t1", 1)

    assert summary["total_tokens"] == 200
    assert summary["estimated_cost"] == 0.01


def test_compute_turn_summary_detects_cache_hit(tmp_path):
    db_path = tmp_path / "telemetry.db"
    _llm(db_path, "t1", 1, 50, 20, 0.0, 5.0, cached_tokens=50)

    summary = compute_turn_summary(db_path, "t1", 1)

    assert summary["cache_hit"] is True


def test_compute_turn_summary_includes_tool_calls(tmp_path):
    db_path = tmp_path / "telemetry.db"
    _llm(db_path, "t1", 1, 100, 100, 0.01, 100.0)
    _tool(db_path, "t1", 1, 50.0)
    _tool(db_path, "t1", 1, 25.0)

    summary = compute_turn_summary(db_path, "t1", 1)

    assert summary["tool_calls"] == 2
    assert summary["latency_ms"] == 175.0  # 100 (llm) + 50 + 25 (tools)


def test_compute_turn_summary_on_empty_turn_returns_zeros(tmp_path):
    db_path = tmp_path / "telemetry.db"
    storage.get_connection(db_path).close()

    summary = compute_turn_summary(db_path, "nonexistent", 1)

    assert summary["total_tokens"] == 0
    assert summary["estimated_cost"] == 0.0
    assert summary["cache_hit"] is False
    assert summary["tool_calls"] == 0


def test_format_turn_summary_includes_tokens_cost_latency_and_cache_status():
    summary = {
        "input_tokens": 3640, "output_tokens": 510, "total_tokens": 4150,
        "estimated_cost": 0.006, "latency_ms": 4000.0, "llm_calls": 3,
        "tool_calls": 0, "cache_hit": False,
    }

    text = format_turn_summary(summary)

    assert "4,150 tokens" in text
    assert "in: 3,640" in text
    assert "out: 510" in text
    assert "$0.0060" in text
    assert "4.0s" in text
    assert "cache: miss" in text
    assert "tool call" not in text


def test_format_turn_summary_pluralizes_tool_calls():
    summary = {
        "input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
        "estimated_cost": 0.0, "latency_ms": 0.0, "llm_calls": 1,
        "tool_calls": 2, "cache_hit": True,
    }

    text = format_turn_summary(summary)

    assert "2 tool calls" in text
    assert "cache: hit" in text
