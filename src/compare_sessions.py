"""
Compares the oldest and newest recorded session (task_id) in telemetry.db
and reports which metrics improved, regressed, or stayed flat. Built for
the "with vs. without optimization" benchmark — run a baseline session,
implement optimizations, run a second session, then run this script.

Usage: python src/compare_sessions.py [--db-path PATH]
"""
import argparse

from rich.console import Console
from rich.table import Table

from telemetry import storage
from telemetry.dashboard import compute_aggregate, compute_comparison, render_comparison, pct_change

# Metrics from compute_comparison() where a smaller value is the win.
IMPROVES_WHEN_LOWER = {"Input tokens", "Output tokens", "Estimated cost", "Tool calls", "Total LLM calls"}
# Metrics where a larger value is the win.
IMPROVES_WHEN_HIGHER = {"Cache hit rate", "Cached tokens"}
# Everything else (e.g. "Turns") is a control variable, not a scored metric.


def find_oldest_and_newest_tasks(db_path=None):
    """Returns (oldest_task_id, newest_task_id) ordered by each task's
    earliest llm_calls timestamp. Returns None if fewer than 2 distinct
    tasks have been recorded."""
    conn = storage.get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT task_id, MIN(timestamp) AS first_seen "
            "FROM llm_calls GROUP BY task_id ORDER BY first_seen"
        ).fetchall()
    finally:
        conn.close()

    if len(rows) < 2:
        return None

    return rows[0][0], rows[-1][0]


def list_sessions(db_path=None):
    """Every recorded session, oldest first, with the summary needed to
    pick two to compare without guessing from a bare task_id: when it
    started, how many turns (questions) it covered, total tokens, cost."""
    conn = storage.get_connection(db_path)
    try:
        rows = conn.execute(
            """
            SELECT task_id, MIN(timestamp) AS started_at,
                   COUNT(DISTINCT turn_number) AS turns,
                   COALESCE(SUM(input_tokens), 0) + COALESCE(SUM(output_tokens), 0) AS total_tokens,
                   COALESCE(SUM(estimated_cost), 0.0) AS cost
            FROM llm_calls
            GROUP BY task_id
            ORDER BY started_at
            """
        ).fetchall()
    finally:
        conn.close()

    return [
        {
            "task_id": row[0], "started_at": row[1], "turns": row[2],
            "total_tokens": row[3], "cost": row[4],
        }
        for row in rows
    ]


def render_session_list(sessions):
    table = Table(title="Recorded sessions")
    table.add_column("#")
    table.add_column("Task ID")
    table.add_column("Started")
    table.add_column("Turns")
    table.add_column("Total tokens")
    table.add_column("Cost")
    for i, session in enumerate(sessions, start=1):
        table.add_row(
            str(i), session["task_id"], session["started_at"],
            str(session["turns"]), str(session["total_tokens"]),
            f"${session['cost']:.4f}",
        )
    return table


def prompt_for_two_sessions(sessions, input_fn=input):
    """Prompts (via input_fn, for testability) for two 1-based indices
    into `sessions` and returns their (task_id_a, task_id_b). Raises
    ValueError — meant to be caught and shown to the user, not a crash —
    on too few sessions, non-numeric input, an out-of-range index, or
    picking the same session twice."""
    if len(sessions) < 2:
        raise ValueError("Need at least 2 recorded sessions to compare.")

    def _read_choice(prompt_text):
        raw = input_fn(prompt_text).strip()
        try:
            index = int(raw)
        except ValueError:
            raise ValueError(f"'{raw}' is not a valid session number.")
        if not (1 <= index <= len(sessions)):
            raise ValueError(f"{index} is out of range (1-{len(sessions)}).")
        return index

    index_a = _read_choice(f"Select session A (1-{len(sessions)}): ")
    index_b = _read_choice(f"Select session B (1-{len(sessions)}): ")

    if index_a == index_b:
        raise ValueError("Session A and session B must be different.")

    return sessions[index_a - 1]["task_id"], sessions[index_b - 1]["task_id"]


def _first_seen(db_path, task_id):
    conn = storage.get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT MIN(timestamp) FROM llm_calls WHERE task_id = ?", (task_id,)
        ).fetchone()
    finally:
        conn.close()
    return row[0]


def _session_label(db_path, task_id, aggregate):
    """One line identifying a session: id, when it started, how many
    questions (turns) it covered — the context a bare task_id lacks, and
    the exact thing that made an earlier 8-question vs. 1-question
    comparison look like a false "improvement"."""
    started = _first_seen(db_path, task_id)
    turns = int(aggregate["avg_turns_per_task"])  # single-task scope: this *is* the task's turn count
    plural = "s" if turns != 1 else ""
    return f"{task_id} (started {started}, {turns} turn{plural})"


def summarize_improvements(rows):
    """rows: compute_comparison()'s output. Returns one human-readable
    line per metric, tagged improved/regressed/unchanged/info."""
    lines = []
    for row in rows:
        metric, before, after, delta = row["metric"], row["session_a"], row["session_b"], row["delta"]
        pct = pct_change(before, after)
        pct_text = f"{pct:+.1f}%" if pct is not None else "n/a"

        if metric in IMPROVES_WHEN_LOWER:
            verdict = "improved" if delta < 0 else ("regressed" if delta > 0 else "unchanged")
        elif metric in IMPROVES_WHEN_HIGHER:
            verdict = "improved" if delta > 0 else ("regressed" if delta < 0 else "unchanged")
        else:
            verdict = "info"

        lines.append(f"{metric}: {before} -> {after} ({pct_text}) [{verdict}]")

    return lines


def render_call_site_breakdown(label, breakdown):
    """breakdown: compute_aggregate()'s "call_site_breakdown" list, for one
    session — shows which of the 3 LLM call sites the tokens/cost went to."""
    table = Table(title=f"{label} — LLM calls by call site")
    table.add_column("Call site")
    table.add_column("Calls")
    table.add_column("Input tokens")
    table.add_column("Output tokens")
    table.add_column("Cost")
    for row in breakdown:
        table.add_row(
            row["call_site"], str(row["call_count"]),
            str(row["input_tokens"]), str(row["output_tokens"]),
            f"${row['estimated_cost']:.4f}",
        )
    return table


def render_tool_breakdown(label, breakdown):
    """breakdown: compute_aggregate()'s "tool_breakdown" list, for one
    session — shows per-tool output size/tokens (not just duration)."""
    table = Table(title=f"{label} — Tool usage")
    table.add_column("Tool")
    table.add_column("Calls")
    table.add_column("Output size (B)")
    table.add_column("Output tokens")
    for row in breakdown:
        table.add_row(
            row["tool_name"], str(row["call_count"]),
            str(row["total_output_size"]), str(row["total_output_tokens"]),
        )
    return table


def run_comparison(db_path, task_id_a, task_id_b):
    """Runs the full comparison + per-session breakdown report for two
    specific task_ids, regardless of how they were chosen (auto
    oldest/newest, or interactively selected)."""
    rows = compute_comparison(db_path, task_id_a, task_id_b)

    agg_a = compute_aggregate(db_path, task_ids=[task_id_a])
    agg_b = compute_aggregate(db_path, task_ids=[task_id_b])

    console = Console()
    console.print(render_comparison(rows, task_id_a, task_id_b))

    print(f"\nSession A: {_session_label(db_path, task_id_a, agg_a)}")
    print(f"Session B: {_session_label(db_path, task_id_b, agg_b)}\n")
    for line in summarize_improvements(rows):
        print(f"  {line}")

    print()
    console.print(render_call_site_breakdown(f"Session {task_id_a}", agg_a["call_site_breakdown"]))
    console.print(render_call_site_breakdown(f"Session {task_id_b}", agg_b["call_site_breakdown"]))

    if agg_a["tool_breakdown"] or agg_b["tool_breakdown"]:
        console.print(render_tool_breakdown(f"Session {task_id_a}", agg_a["tool_breakdown"]))
        console.print(render_tool_breakdown(f"Session {task_id_b}", agg_b["tool_breakdown"]))


def main(db_path=None, select=False, input_fn=input):
    if select:
        sessions = list_sessions(db_path)
        if len(sessions) < 2:
            print("Need at least 2 recorded sessions (task_ids) to compare — found fewer than that.")
            return

        Console().print(render_session_list(sessions))
        try:
            task_id_a, task_id_b = prompt_for_two_sessions(sessions, input_fn=input_fn)
        except ValueError as e:
            print(f"❌ {e}")
            return

        run_comparison(db_path, task_id_a, task_id_b)
        return

    tasks = find_oldest_and_newest_tasks(db_path)
    if tasks is None:
        print("Need at least 2 recorded sessions (task_ids) to compare — found fewer than that.")
        return

    run_comparison(db_path, *tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default=None)
    parser.add_argument(
        "--select", action="store_true",
        help="Pick which two sessions to compare from a numbered list, instead of auto oldest-vs-newest",
    )
    args = parser.parse_args()
    main(args.db_path, select=args.select)
