from rich.table import Table

from telemetry import storage


def fetch_timeline(db_path, task_id):
    """Every llm_calls/tool_calls row for `task_id`, ordered by
    (turn_number, timestamp)."""
    conn = storage.get_connection(db_path)
    try:
        llm_rows = conn.execute(
            """
            SELECT turn_number, timestamp, call_site, input_tokens,
                   output_tokens, cached_tokens, latency_ms, estimated_cost
            FROM llm_calls WHERE task_id = ?
            """,
            (task_id,),
        ).fetchall()

        tool_rows = conn.execute(
            """
            SELECT turn_number, timestamp, tool_name, input_size,
                   output_size, output_tokens, duration_ms
            FROM tool_calls WHERE task_id = ?
            """,
            (task_id,),
        ).fetchall()
    finally:
        conn.close()

    events = [
        {
            "turn_number": row[0], "timestamp": row[1], "type": "llm", "label": row[2],
            "input_tokens": row[3], "output_tokens": row[4], "cached_tokens": row[5],
            "latency_ms": row[6], "estimated_cost": row[7],
        }
        for row in llm_rows
    ] + [
        {
            "turn_number": row[0], "timestamp": row[1], "type": "tool", "label": row[2],
            "input_size": row[3], "output_size": row[4], "output_tokens": row[5],
            "duration_ms": row[6],
        }
        for row in tool_rows
    ]

    events.sort(key=lambda e: (e["turn_number"], e["timestamp"]))
    return events


def render_timeline(task_id, events):
    table = Table(title=f"Timeline — task {task_id}")
    table.add_column("Turn")
    table.add_column("Type")
    table.add_column("Detail")
    table.add_column("Tokens/Size")
    table.add_column("Latency (ms)")
    table.add_column("Cost")

    for event in events:
        if event["type"] == "llm":
            tokens = f"{event['input_tokens']}in/{event['output_tokens']}out"
            if event["cached_tokens"] > 0:
                tokens += " (cached)"
            table.add_row(
                f"Turn {event['turn_number']}", "LLM", event["label"], tokens,
                f"{event['latency_ms']:.1f}", f"${event['estimated_cost']:.6f}",
            )
        else:
            tokens = f"{event['output_tokens']} tok / {event['output_size']}B"
            table.add_row(
                f"Turn {event['turn_number']}", "Tool", event["label"], tokens,
                f"{event['duration_ms']:.1f}", "-",
            )

    return table
