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


def test_compute_aggregate_tool_breakdown_includes_output_size_and_tokens(tmp_path):
    db_path = tmp_path / "telemetry.db"
    _seed(db_path, "t1", 1, 10, 10, 0, 0.0)
    storage.insert_tool_call(
        agent_id="a", task_id="t1", turn_number=1, tool_name="read_document",
        input_size=10, output_size=100, output_tokens=25, duration_ms=50.0, db_path=db_path,
    )
    storage.insert_tool_call(
        agent_id="a", task_id="t1", turn_number=1, tool_name="read_document",
        input_size=10, output_size=50, output_tokens=12, duration_ms=30.0, db_path=db_path,
    )

    result = compute_aggregate(db_path)

    row = result["tool_breakdown"][0]
    assert row["tool_name"] == "read_document"
    assert row["total_output_size"] == 150
    assert row["total_output_tokens"] == 37


def test_compute_aggregate_returns_total_llm_calls(tmp_path):
    db_path = tmp_path / "telemetry.db"
    _seed(db_path, "t1", 1, 100, 50, 0, 0.03, call_site="ask_llm")
    _seed(db_path, "t1", 1, 20, 10, 0, 0.01, call_site="mcp_decision")
    _seed(db_path, "t1", 1, 5, 5, 0, 0.001, call_site="query_expansion")

    result = compute_aggregate(db_path)

    assert result["total_llm_calls"] == 3


def test_compute_aggregate_call_site_breakdown_groups_and_sums_correctly(tmp_path):
    db_path = tmp_path / "telemetry.db"
    _seed(db_path, "t1", 1, 100, 50, 0, 0.03, call_site="ask_llm")
    _seed(db_path, "t1", 2, 200, 100, 0, 0.06, call_site="ask_llm")
    _seed(db_path, "t1", 1, 20, 10, 0, 0.01, call_site="mcp_decision")

    result = compute_aggregate(db_path)
    by_call_site = {row["call_site"]: row for row in result["call_site_breakdown"]}

    assert by_call_site["ask_llm"]["call_count"] == 2
    assert by_call_site["ask_llm"]["input_tokens"] == 300
    assert by_call_site["ask_llm"]["output_tokens"] == 150
    assert by_call_site["ask_llm"]["estimated_cost"] == pytest.approx(0.09)
    assert by_call_site["mcp_decision"]["call_count"] == 1
    assert by_call_site["mcp_decision"]["input_tokens"] == 20


def test_compute_aggregate_call_site_breakdown_sorted_by_cost_desc(tmp_path):
    db_path = tmp_path / "telemetry.db"
    _seed(db_path, "t1", 1, 10, 10, 0, 0.001, call_site="query_expansion")
    _seed(db_path, "t1", 1, 100, 100, 0, 0.05, call_site="ask_llm")

    result = compute_aggregate(db_path)

    assert [row["call_site"] for row in result["call_site_breakdown"]] == ["ask_llm", "query_expansion"]


def test_compute_aggregate_on_empty_db_has_empty_call_site_breakdown_and_zero_llm_calls(tmp_path):
    db_path = tmp_path / "telemetry.db"
    storage.get_connection(db_path).close()

    result = compute_aggregate(db_path)

    assert result["call_site_breakdown"] == []
    assert result["total_llm_calls"] == 0


def test_compute_comparison_delta_increase_decrease_and_unchanged(tmp_path):
    db_path = tmp_path / "telemetry.db"
    _seed(db_path, "t1", 1, 100, 100, 0, 0.10)
    _seed(db_path, "t2", 1, 150, 100, 0, 0.05)

    rows = compute_comparison(db_path, "t1", "t2")
    by_metric = {row["metric"]: row for row in rows}

    assert by_metric["Input tokens"]["delta"] == 50                          # increase
    assert by_metric["Estimated cost"]["delta"] == pytest.approx(-0.05)      # decrease
    assert by_metric["Output tokens"]["delta"] == 0                         # unchanged


def test_compute_comparison_includes_total_llm_calls(tmp_path):
    db_path = tmp_path / "telemetry.db"
    _seed(db_path, "t1", 1, 100, 100, 0, 0.10)
    _seed(db_path, "t1", 2, 100, 100, 0, 0.10)
    _seed(db_path, "t2", 1, 100, 100, 0, 0.05)

    rows = compute_comparison(db_path, "t1", "t2")
    by_metric = {row["metric"]: row for row in rows}

    assert by_metric["Total LLM calls"]["session_a"] == 2
    assert by_metric["Total LLM calls"]["session_b"] == 1
    assert by_metric["Total LLM calls"]["delta"] == -1


def test_render_dashboard_includes_key_metrics_as_text():
    aggregate = {
        "tasks_completed": 3, "total_input_tokens": 100, "total_output_tokens": 50,
        "total_cached_tokens": 10, "estimated_cost": 0.05, "avg_tokens_per_task": 50.0,
        "avg_turns_per_task": 2.0, "avg_tool_calls_per_task": 1.0, "cache_hit_rate": 0.25,
        "total_llm_calls": 4, "call_site_breakdown": [], "tool_breakdown": [],
    }

    console = Console(record=True, width=120)
    console.print(render_dashboard(aggregate))
    text = console.export_text()

    assert "Tasks completed" in text
    assert "3" in text
    assert "25.0%" in text
    assert "Total LLM calls" in text


def test_render_dashboard_includes_call_site_breakdown_table():
    aggregate = {
        "tasks_completed": 1, "total_input_tokens": 100, "total_output_tokens": 50,
        "total_cached_tokens": 0, "estimated_cost": 0.05, "avg_tokens_per_task": 150.0,
        "avg_turns_per_task": 1.0, "avg_tool_calls_per_task": 0.0, "cache_hit_rate": 0.0,
        "total_llm_calls": 1,
        "call_site_breakdown": [
            {"call_site": "ask_llm", "call_count": 1, "input_tokens": 100, "output_tokens": 50, "estimated_cost": 0.05},
        ],
        "tool_breakdown": [],
    }

    console = Console(record=True, width=120)
    console.print(render_dashboard(aggregate))
    text = console.export_text()

    assert "ask_llm" in text
    assert "LLM calls by call site" in text


def test_render_dashboard_tool_breakdown_includes_output_size_and_tokens():
    aggregate = {
        "tasks_completed": 1, "total_input_tokens": 0, "total_output_tokens": 0,
        "total_cached_tokens": 0, "estimated_cost": 0.0, "avg_tokens_per_task": 0.0,
        "avg_turns_per_task": 1.0, "avg_tool_calls_per_task": 1.0, "cache_hit_rate": 0.0,
        "total_llm_calls": 0, "call_site_breakdown": [],
        "tool_breakdown": [
            {
                "tool_name": "read_document", "call_count": 2,
                "total_duration_ms": 80.0, "avg_duration_ms": 40.0,
                "total_output_size": 150, "total_output_tokens": 37,
            },
        ],
    }

    console = Console(record=True, width=120)
    console.print(render_dashboard(aggregate))
    text = console.export_text()

    assert "150" in text
    assert "37" in text


def test_render_comparison_includes_session_ids_and_delta_column():
    rows = [{"metric": "Input tokens", "session_a": 100, "session_b": 150, "delta": 50}]

    console = Console(record=True, width=120)
    console.print(render_comparison(rows, "t1", "t2"))
    text = console.export_text()

    assert "t1" in text
    assert "t2" in text
    assert "Input tokens" in text
