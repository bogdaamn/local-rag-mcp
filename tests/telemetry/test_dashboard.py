import pytest
from rich.console import Console

from telemetry import storage
from telemetry.dashboard import (
    compute_aggregate,
    compute_comparison,
    render_dashboard,
    render_comparison,
)


def _seed(db_path, task_id, turn_number, input_tokens, output_tokens, cached_tokens, cost, call_site="ask_llm"):
    storage.insert_llm_call(
        agent_id="a", task_id=task_id, turn_number=turn_number, model="qwen3:0.6b",
        input_tokens=input_tokens, output_tokens=output_tokens, cached_tokens=cached_tokens,
        reasoning_tokens=0, latency_ms=10.0, estimated_cost=cost, call_site=call_site,
        db_path=db_path,
    )


def test_compute_aggregate_on_empty_db_returns_zeros(tmp_path):
    db_path = tmp_path / "telemetry.db"
    storage.get_connection(db_path).close()

    result = compute_aggregate(db_path)

    assert result["tasks_completed"] == 0
    assert result["total_input_tokens"] == 0
    assert result["avg_tokens_per_task"] == 0.0
    assert result["cache_hit_rate"] == 0.0
    assert result["tool_breakdown"] == []


def test_compute_aggregate_single_task(tmp_path):
    db_path = tmp_path / "telemetry.db"
    _seed(db_path, "t1", 1, 100, 50, 0, 0.03)
    _seed(db_path, "t1", 1, 20, 10, 20, 0.0, call_site="query_expansion")  # cache hit

    result = compute_aggregate(db_path)

    assert result["tasks_completed"] == 1
    assert result["total_input_tokens"] == 120
    assert result["total_output_tokens"] == 60
    assert result["total_cached_tokens"] == 20
    assert result["estimated_cost"] == 0.03
    assert result["avg_tokens_per_task"] == 180.0
    assert result["avg_turns_per_task"] == 1.0
    assert result["cache_hit_rate"] == 0.5  # 1 of 2 llm_calls rows was a hit


def test_compute_aggregate_multi_task_scoped_by_task_ids(tmp_path):
    db_path = tmp_path / "telemetry.db"
    _seed(db_path, "t1", 1, 100, 50, 0, 0.03)
    _seed(db_path, "t2", 1, 200, 100, 0, 0.06)
    _seed(db_path, "t3", 1, 999, 999, 0, 9.99)  # excluded by task_ids filter

    result = compute_aggregate(db_path, task_ids=["t1", "t2"])

    assert result["tasks_completed"] == 2
    assert result["total_input_tokens"] == 300
    assert result["estimated_cost"] == 0.09


def test_compute_aggregate_tool_breakdown_sorted_by_total_duration_desc(tmp_path):
    db_path = tmp_path / "telemetry.db"
    _seed(db_path, "t1", 1, 10, 10, 0, 0.0)
    storage.insert_tool_call(
        agent_id="a", task_id="t1", turn_number=1, tool_name="read_document",
        input_size=10, output_size=100, output_tokens=25, duration_ms=50.0, db_path=db_path,
    )
    storage.insert_tool_call(
        agent_id="a", task_id="t1", turn_number=1, tool_name="list_documents",
        input_size=2, output_size=20, output_tokens=5, duration_ms=200.0, db_path=db_path,
    )

    result = compute_aggregate(db_path)

    assert [row["tool_name"] for row in result["tool_breakdown"]] == ["list_documents", "read_document"]
    assert result["avg_tool_calls_per_task"] == 2.0


def test_compute_comparison_delta_increase_decrease_and_unchanged(tmp_path):
    db_path = tmp_path / "telemetry.db"
    _seed(db_path, "t1", 1, 100, 100, 0, 0.10)
    _seed(db_path, "t2", 1, 150, 100, 0, 0.05)

    rows = compute_comparison(db_path, "t1", "t2")
    by_metric = {row["metric"]: row for row in rows}

    assert by_metric["Input tokens"]["delta"] == 50                          # increase
    assert by_metric["Estimated cost"]["delta"] == pytest.approx(-0.05)      # decrease
    assert by_metric["Output tokens"]["delta"] == 0                         # unchanged


def test_render_dashboard_includes_key_metrics_as_text():
    aggregate = {
        "tasks_completed": 3, "total_input_tokens": 100, "total_output_tokens": 50,
        "total_cached_tokens": 10, "estimated_cost": 0.05, "avg_tokens_per_task": 50.0,
        "avg_turns_per_task": 2.0, "avg_tool_calls_per_task": 1.0, "cache_hit_rate": 0.25,
        "tool_breakdown": [],
    }

    console = Console(record=True, width=120)
    console.print(render_dashboard(aggregate))
    text = console.export_text()

    assert "Tasks completed" in text
    assert "3" in text
    assert "25.0%" in text


def test_render_comparison_includes_session_ids_and_delta_column():
    rows = [{"metric": "Input tokens", "session_a": 100, "session_b": 150, "delta": 50}]

    console = Console(record=True, width=120)
    console.print(render_comparison(rows, "t1", "t2"))
    text = console.export_text()

    assert "t1" in text
    assert "t2" in text
    assert "Input tokens" in text
