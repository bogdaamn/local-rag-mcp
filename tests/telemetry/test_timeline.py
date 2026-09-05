from rich.console import Console

from telemetry import storage
from telemetry.timeline import fetch_timeline, render_timeline


def test_fetch_timeline_orders_by_turn_then_timestamp(tmp_path):
    db_path = tmp_path / "telemetry.db"
    storage.insert_llm_call(
        agent_id="a", task_id="t1", turn_number=2, model="m", input_tokens=1,
        output_tokens=1, cached_tokens=0, reasoning_tokens=0, latency_ms=1.0,
        estimated_cost=0.0, call_site="ask_llm", timestamp="2026-09-05T10:00:02",
        db_path=db_path,
    )
    storage.insert_llm_call(
        agent_id="a", task_id="t1", turn_number=1, model="m", input_tokens=2,
        output_tokens=2, cached_tokens=0, reasoning_tokens=0, latency_ms=1.0,
        estimated_cost=0.0, call_site="mcp_decision", timestamp="2026-09-05T10:00:00",
        db_path=db_path,
    )
    storage.insert_tool_call(
        agent_id="a", task_id="t1", turn_number=1, tool_name="list_documents",
        input_size=2, output_size=10, output_tokens=3, duration_ms=5.0,
        timestamp="2026-09-05T10:00:01", db_path=db_path,
    )

    events = fetch_timeline(db_path, "t1")

    assert [e["turn_number"] for e in events] == [1, 1, 2]
    assert [e["type"] for e in events] == ["llm", "tool", "llm"]
    assert events[0]["label"] == "mcp_decision"
    assert events[1]["label"] == "list_documents"


def test_fetch_timeline_only_returns_the_requested_task(tmp_path):
    db_path = tmp_path / "telemetry.db"
    storage.insert_llm_call(
        agent_id="a", task_id="t1", turn_number=1, model="m", input_tokens=1,
        output_tokens=1, cached_tokens=0, reasoning_tokens=0, latency_ms=1.0,
        estimated_cost=0.0, call_site="ask_llm", db_path=db_path,
    )
    storage.insert_llm_call(
        agent_id="a", task_id="t2", turn_number=1, model="m", input_tokens=1,
        output_tokens=1, cached_tokens=0, reasoning_tokens=0, latency_ms=1.0,
        estimated_cost=0.0, call_site="ask_llm", db_path=db_path,
    )

    events = fetch_timeline(db_path, "t1")

    assert len(events) == 1


def test_render_timeline_lists_turn_and_type_for_each_event():
    events = [
        {
            "turn_number": 1, "timestamp": "t", "type": "llm", "label": "ask_llm",
            "input_tokens": 10, "output_tokens": 5, "cached_tokens": 0,
            "latency_ms": 12.3, "estimated_cost": 0.0001,
        },
        {
            "turn_number": 1, "timestamp": "t", "type": "tool", "label": "list_documents",
            "input_size": 2, "output_size": 20, "output_tokens": 5, "duration_ms": 8.0,
        },
    ]

    console = Console(record=True, width=120)
    console.print(render_timeline("t1", events))
    text = console.export_text()

    assert "Turn 1" in text
    assert "ask_llm" in text
    assert "list_documents" in text
