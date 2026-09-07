from rich.console import Group
from rich.table import Table

from telemetry import storage


def _task_filter_clause(task_ids):
    if not task_ids:
        return "", []
    placeholders = ",".join("?" for _ in task_ids)
    return f"WHERE task_id IN ({placeholders})", list(task_ids)


def _safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def compute_aggregate(db_path, task_ids=None):
    """Aggregate stats across all history, or scoped to `task_ids` if given."""
    where_clause, params = _task_filter_clause(task_ids)

    conn = storage.get_connection(db_path)
    try:
        llm_row = conn.execute(
            f"""
            SELECT
                COUNT(DISTINCT task_id),
                COALESCE(SUM(input_tokens), 0),
                COALESCE(SUM(output_tokens), 0),
                COALESCE(SUM(cached_tokens), 0),
                COALESCE(SUM(estimated_cost), 0.0),
                COUNT(*),
                COALESCE(SUM(CASE WHEN cached_tokens > 0 THEN 1 ELSE 0 END), 0)
            FROM llm_calls {where_clause}
            """,
            params,
        ).fetchone()

        turns_row = conn.execute(
            f"""
            SELECT COUNT(*) FROM (
                SELECT DISTINCT task_id, turn_number FROM llm_calls {where_clause}
            )
            """,
            params,
        ).fetchone()

        total_tool_calls = conn.execute(
            f"SELECT COUNT(*) FROM tool_calls {where_clause}", params
        ).fetchone()[0]

        tool_breakdown = conn.execute(
            f"""
            SELECT tool_name, COUNT(*), SUM(duration_ms), AVG(duration_ms)
            FROM tool_calls {where_clause}
            GROUP BY tool_name
            ORDER BY SUM(duration_ms) DESC
            """,
            params,
        ).fetchall()
    finally:
        conn.close()

    (
        tasks_completed, total_input_tokens, total_output_tokens,
        total_cached_tokens, total_estimated_cost, total_llm_calls, cache_hits,
    ) = llm_row
    total_turns = turns_row[0] or 0

    return {
        "tasks_completed": tasks_completed,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_cached_tokens": total_cached_tokens,
        "estimated_cost": total_estimated_cost,
        "avg_tokens_per_task": _safe_div(total_input_tokens + total_output_tokens, tasks_completed),
        "avg_turns_per_task": _safe_div(total_turns, tasks_completed),
        "avg_tool_calls_per_task": _safe_div(total_tool_calls, tasks_completed),
        "cache_hit_rate": _safe_div(cache_hits, total_llm_calls),
        "tool_breakdown": [
            {
                "tool_name": row[0], "call_count": row[1],
                "total_duration_ms": row[2], "avg_duration_ms": row[3],
            }
            for row in tool_breakdown
        ],
    }


def compute_comparison(db_path, task_id_a, task_id_b):
    """Exactly-2-task delta comparison: [{"metric", "session_a", "session_b", "delta"}, ...]."""
    agg_a = compute_aggregate(db_path, task_ids=[task_id_a])
    agg_b = compute_aggregate(db_path, task_ids=[task_id_b])

    metrics = [
        ("Input tokens", "total_input_tokens"),
        ("Output tokens", "total_output_tokens"),
        ("Cached tokens", "total_cached_tokens"),
        ("Estimated cost", "estimated_cost"),
        ("Turns", "avg_turns_per_task"),
        ("Tool calls", "avg_tool_calls_per_task"),
        ("Cache hit rate", "cache_hit_rate"),
    ]

    return [
        {
            "metric": label,
            "session_a": agg_a[key],
            "session_b": agg_b[key],
            "delta": agg_b[key] - agg_a[key],
        }
        for label, key in metrics
    ]


def render_dashboard(aggregate):
    table = Table(title="AI Agent — Aggregate Stats")
    table.add_column("Metric")
    table.add_column("Value")

    table.add_row("Tasks completed", str(aggregate["tasks_completed"]))
    table.add_row("Total input tokens", str(aggregate["total_input_tokens"]))
    table.add_row("Total output tokens", str(aggregate["total_output_tokens"]))
    table.add_row("Total cached tokens", str(aggregate["total_cached_tokens"]))
    table.add_row("Estimated cost", f"${aggregate['estimated_cost']:.4f}")
    table.add_row("Avg tokens/task", f"{aggregate['avg_tokens_per_task']:.1f}")
    table.add_row("Avg turns/task", f"{aggregate['avg_turns_per_task']:.1f}")
    table.add_row("Avg tool calls/task", f"{aggregate['avg_tool_calls_per_task']:.1f}")
    table.add_row("Cache hit rate", f"{aggregate['cache_hit_rate'] * 100:.1f}%")

    if not aggregate["tool_breakdown"]:
        return table

    breakdown_table = Table(title="Tool usage breakdown")
    breakdown_table.add_column("Tool")
    breakdown_table.add_column("Calls")
    breakdown_table.add_column("Total duration (ms)")
    breakdown_table.add_column("Avg duration (ms)")
    for row in aggregate["tool_breakdown"]:
        breakdown_table.add_row(
            row["tool_name"], str(row["call_count"]),
            f"{row['total_duration_ms']:.1f}", f"{row['avg_duration_ms']:.1f}",
        )

    return Group(table, breakdown_table)


def _format_metric_value(metric, value):
    if metric == "Estimated cost":
        return f"${value:.4f}"
    if metric == "Cache hit rate":
        return f"{value * 100:.1f}%"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def render_comparison(rows, task_id_a, task_id_b):
    table = Table(title="Session comparison")
    table.add_column("Metric")
    table.add_column(f"Session {task_id_a}")
    table.add_column(f"Session {task_id_b}")
    table.add_column("Δ")

    for row in rows:
        delta = row["delta"]
        sign = "+" if delta > 0 else ""
        table.add_row(
            row["metric"],
            _format_metric_value(row["metric"], row["session_a"]),
            _format_metric_value(row["metric"], row["session_b"]),
            f"{sign}{_format_metric_value(row['metric'], delta)}",
        )

    return table
