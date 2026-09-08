import pytest
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


def test_main_labels_each_session_with_start_time_and_turn_count(tmp_path, capsys):
    db_path = tmp_path / "telemetry.db"
    _seed(db_path, "oldest", "2026-09-06T10:00:00")
    _seed(db_path, "newest", "2026-09-06T14:00:00")
    storage.insert_llm_call(
        agent_id="a", task_id="newest", turn_number=2, model="qwen3:0.6b",
        input_tokens=1, output_tokens=1, cached_tokens=0, reasoning_tokens=0,
        latency_ms=1.0, estimated_cost=0.0, call_site="ask_llm",
        timestamp="2026-09-06T14:05:00", db_path=db_path,
    )

    compare_sessions.main(db_path)

    out = capsys.readouterr().out
    assert "2026-09-06T10:00:00" in out
    assert "1 turn" in out          # oldest: single seeded call, 1 turn
    assert "2026-09-06T14:00:00" in out
    assert "2 turns" in out         # newest: turn 1 and turn 2


def test_list_sessions_orders_oldest_first_with_summary_fields(tmp_path):
    db_path = tmp_path / "telemetry.db"
    _seed(db_path, "b", "2026-09-06T14:00:00", input_tokens=10, output_tokens=5, cost=0.02)
    _seed(db_path, "a", "2026-09-06T10:00:00", input_tokens=100, output_tokens=50, cost=0.03)

    sessions = compare_sessions.list_sessions(db_path)

    assert [s["task_id"] for s in sessions] == ["a", "b"]
    assert sessions[0]["started_at"] == "2026-09-06T10:00:00"
    assert sessions[0]["turns"] == 1
    assert sessions[0]["total_tokens"] == 150
    assert sessions[0]["cost"] == pytest.approx(0.03)


def test_list_sessions_on_empty_db_returns_empty_list(tmp_path):
    db_path = tmp_path / "telemetry.db"
    storage.get_connection(db_path).close()

    assert compare_sessions.list_sessions(db_path) == []


def test_render_session_list_shows_numbered_rows():
    sessions = [
        {"task_id": "abc123", "started_at": "2026-09-06T10:00:00", "turns": 8, "total_tokens": 100, "cost": 0.01},
        {"task_id": "def456", "started_at": "2026-09-06T14:00:00", "turns": 1, "total_tokens": 10, "cost": 0.001},
    ]

    console = Console(record=True, width=120)
    console.print(compare_sessions.render_session_list(sessions))
    text = console.export_text()

    assert "abc123" in text
    assert "def456" in text
    assert "8" in text


def test_prompt_for_two_sessions_returns_task_ids_for_valid_input():
    sessions = [{"task_id": "a"}, {"task_id": "b"}, {"task_id": "c"}]
    answers = iter(["1", "3"])

    result = compare_sessions.prompt_for_two_sessions(sessions, input_fn=lambda _: next(answers))

    assert result == ("a", "c")


def test_prompt_for_two_sessions_rejects_non_numeric_input():
    sessions = [{"task_id": "a"}, {"task_id": "b"}]

    with pytest.raises(ValueError, match="not a valid session number"):
        compare_sessions.prompt_for_two_sessions(sessions, input_fn=lambda _: "nope")


def test_prompt_for_two_sessions_rejects_out_of_range_index():
    sessions = [{"task_id": "a"}, {"task_id": "b"}]

    with pytest.raises(ValueError, match="out of range"):
        compare_sessions.prompt_for_two_sessions(sessions, input_fn=lambda _: "5")


def test_prompt_for_two_sessions_rejects_picking_the_same_session_twice():
    sessions = [{"task_id": "a"}, {"task_id": "b"}]
    answers = iter(["1", "1"])

    with pytest.raises(ValueError, match="must be different"):
        compare_sessions.prompt_for_two_sessions(sessions, input_fn=lambda _: next(answers))


def test_prompt_for_two_sessions_rejects_fewer_than_two_sessions():
    with pytest.raises(ValueError, match="at least 2"):
        compare_sessions.prompt_for_two_sessions([{"task_id": "a"}], input_fn=lambda _: "1")


def test_main_select_mode_runs_comparison_for_chosen_sessions(tmp_path, capsys):
    db_path = tmp_path / "telemetry.db"
    _seed(db_path, "a", "2026-09-06T10:00:00", call_site="mcp_decision")
    _seed(db_path, "b", "2026-09-06T12:00:00", call_site="ask_llm")
    _seed(db_path, "c", "2026-09-06T14:00:00", call_site="query_expansion")
    answers = iter(["1", "3"])  # pick session "a" and "c", skip "b"

    compare_sessions.main(db_path, select=True, input_fn=lambda _: next(answers))

    out = capsys.readouterr().out
    assert "Recorded sessions" in out
    assert "mcp_decision" in out
    assert "query_expansion" in out
    assert "ask_llm" not in out  # session "b" was not selected


def test_main_select_mode_with_fewer_than_two_sessions_reports_and_does_not_prompt(tmp_path, capsys):
    db_path = tmp_path / "telemetry.db"
    _seed(db_path, "only-one", "2026-09-06T10:00:00")

    def _fail_if_called(_):
        raise AssertionError("should not prompt with fewer than 2 sessions")

    compare_sessions.main(db_path, select=True, input_fn=_fail_if_called)

    assert "Need at least 2 recorded sessions" in capsys.readouterr().out


def test_main_select_mode_reports_invalid_choice_without_raising(tmp_path, capsys):
    db_path = tmp_path / "telemetry.db"
    _seed(db_path, "a", "2026-09-06T10:00:00")
    _seed(db_path, "b", "2026-09-06T14:00:00")

    compare_sessions.main(db_path, select=True, input_fn=lambda _: "9")

    assert "out of range" in capsys.readouterr().out
