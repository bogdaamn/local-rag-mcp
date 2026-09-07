from telemetry import storage


def compute_turn_summary(db_path, task_id, turn_number):
    """Aggregate one turn's llm_calls + tool_calls rows into a small dict
    of user-facing stats, for printing right after an answer."""
    conn = storage.get_connection(db_path)
    try:
        llm_row = conn.execute(
            """
            SELECT
                COALESCE(SUM(input_tokens), 0),
                COALESCE(SUM(output_tokens), 0),
                COALESCE(SUM(estimated_cost), 0.0),
                COALESCE(SUM(latency_ms), 0.0),
                COALESCE(SUM(CASE WHEN cached_tokens > 0 THEN 1 ELSE 0 END), 0),
                COUNT(*)
            FROM llm_calls WHERE task_id = ? AND turn_number = ?
            """,
            (task_id, turn_number),
        ).fetchone()

        tool_row = conn.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(duration_ms), 0.0)
            FROM tool_calls WHERE task_id = ? AND turn_number = ?
            """,
            (task_id, turn_number),
        ).fetchone()
    finally:
        conn.close()

    input_tokens, output_tokens, cost, llm_latency_ms, cache_hits, llm_calls = llm_row
    tool_calls, tool_latency_ms = tool_row

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "estimated_cost": cost,
        "latency_ms": llm_latency_ms + tool_latency_ms,
        "llm_calls": llm_calls,
        "tool_calls": tool_calls,
        "cache_hit": cache_hits > 0,
    }


def format_turn_summary(summary):
    """One-line, human-readable rendering of compute_turn_summary()'s dict."""
    cache_label = "hit" if summary["cache_hit"] else "miss"
    parts = [
        f"{summary['total_tokens']:,} tokens (in: {summary['input_tokens']:,} / out: {summary['output_tokens']:,})",
        f"${summary['estimated_cost']:.4f}",
        f"{summary['latency_ms'] / 1000:.1f}s",
        f"cache: {cache_label}",
    ]
    if summary["tool_calls"]:
        plural = "s" if summary["tool_calls"] != 1 else ""
        parts.append(f"{summary['tool_calls']} tool call{plural}")
    return " · ".join(parts)
