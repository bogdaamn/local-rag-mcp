from rich.console import Console

import compare_sessions
from telemetry import storage


def _seed(db_path, task_id, timestamp, input_tokens=10, output_tokens=10, cost=0.01, call_site="ask_llm"):
    storage.insert_llm_call(
        agent_id="a", task_id=task_id, turn_number=1, model="qwen3:0.6b",
        input_tokens=input_tokens, output_tokens=output_tokens, cached_tokens=0,
        reasoning_tokens=0, latency_ms=1.0, estimated_cost=cost, call_site=call_site,
        timestamp=timestamp, db_path=db_path,
    )


def test_find_oldest_and_newest_tasks_orders_by_first_timestamp(tmp_path):
    db_path = tmp_path / "telemetry.db"
    _seed(db_path, "middle", "2026-09-06T12:00:00")
    _seed(db_path, "oldest", "2026-09-06T10:00:00")
    _seed(db_path, "newest", "2026-09-06T14:00:00")

    result = compare_sessions.find_oldest_and_newest_tasks(db_path)

    assert result == ("oldest", "newest")


def test_find_oldest_and_newest_tasks_returns_none_with_fewer_than_two(tmp_path):
    db_path = tmp_path / "telemetry.db"
    _seed(db_path, "only-one", "2026-09-06T10:00:00")

    assert compare_sessions.find_oldest_and_newest_tasks(db_path) is None


def test_find_oldest_and_newest_tasks_returns_none_on_empty_db(tmp_path):
    db_path = tmp_path / "telemetry.db"
    storage.get_connection(db_path).close()

    assert compare_sessions.find_oldest_and_newest_tasks(db_path) is None


def test_summarize_improvements_flags_lower_is_better_metric():
    rows = [{"metric": "Input tokens", "session_a": 100, "session_b": 60, "delta": -40}]

    lines = compare_sessions.summarize_improvements(rows)

    assert lines == ["Input tokens: 100 -> 60 (-40.0%) [improved]"]


def test_summarize_improvements_flags_lower_is_better_regression():
    rows = [{"metric": "Estimated cost", "session_a": 0.01, "session_b": 0.02, "delta": 0.01}]

    lines = compare_sessions.summarize_improvements(rows)

    assert "[regressed]" in lines[0]


def test_summarize_improvements_flags_higher_is_better_metric():
    rows = [{"metric": "Cache hit rate", "session_a": 0.0, "session_b": 0.5, "delta": 0.5}]

    lines = compare_sessions.summarize_improvements(rows)

    assert lines[0].endswith("[improved]")


def test_summarize_improvements_treats_unlisted_metric_as_info():
    rows = [{"metric": "Turns", "session_a": 8, "session_b": 8, "delta": 0}]

    lines = compare_sessions.summarize_improvements(rows)

    assert lines[0].endswith("[info]")


def test_summarize_improvements_handles_zero_before_value_without_raising():
    rows = [{"metric": "Tool calls", "session_a": 0, "session_b": 2, "delta": 2}]

    lines = compare_sessions.summarize_improvements(rows)

    assert "n/a" in lines[0]


def test_summarize_improvements_flags_total_llm_calls_as_lower_is_better():
    rows = [{"metric": "Total LLM calls", "session_a": 24, "session_b": 15, "delta": -9}]

    lines = compare_sessions.summarize_improvements(rows)

    assert lines[0].endswith("[improved]")


def test_main_reports_when_fewer_than_two_sessions(tmp_path, capsys):
    db_path = tmp_path / "telemetry.db"
    _seed(db_path, "only-one", "2026-09-06T10:00:00")

    compare_sessions.main(db_path)

    assert "Need at least 2 recorded sessions" in capsys.readouterr().out


def test_render_call_site_breakdown_shows_call_site_and_tokens():
    breakdown = [
        {"call_site": "ask_llm", "call_count": 1, "input_tokens": 100, "output_tokens": 50, "estimated_cost": 0.05},
    ]

    console = Console(record=True, width=120)
    console.print(compare_sessions.render_call_site_breakdown("Session A", breakdown))
    text = console.export_text()

    assert "ask_llm" in text
    assert "100" in text
    assert "Session A" in text


def test_render_tool_breakdown_shows_output_size_and_tokens():
    breakdown = [
        {"tool_name": "read_document", "call_count": 1, "total_output_size": 150, "total_output_tokens": 37},
    ]

    console = Console(record=True, width=120)
    console.print(compare_sessions.render_tool_breakdown("Session A", breakdown))
    text = console.export_text()

    assert "read_document" in text
    assert "150" in text
    assert "37" in text


def test_main_prints_per_session_breakdowns(tmp_path, capsys):
    db_path = tmp_path / "telemetry.db"
    _seed(db_path, "oldest", "2026-09-06T10:00:00", call_site="mcp_decision")
    _seed(db_path, "newest", "2026-09-06T14:00:00", call_site="ask_llm")

    compare_sessions.main(db_path)

    out = capsys.readouterr().out
    assert "mcp_decision" in out
    assert "ask_llm" in out
