"""
Compares the oldest and newest recorded session (task_id) in telemetry.db
and reports which metrics improved, regressed, or stayed flat. Built for
the "with vs. without optimization" benchmark — run a baseline session,
implement optimizations, run a second session, then run this script.

Usage: python src/compare_sessions.py [--db-path PATH]
"""
import argparse

from rich.console import Console

from telemetry import storage
from telemetry.dashboard import compute_comparison, render_comparison

# Metrics from compute_comparison() where a smaller value is the win.
IMPROVES_WHEN_LOWER = {"Input tokens", "Output tokens", "Estimated cost", "Tool calls"}
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


def _pct_change(before, after):
    if before == 0:
        return None
    return (after - before) / abs(before) * 100


def summarize_improvements(rows):
    """rows: compute_comparison()'s output. Returns one human-readable
    line per metric, tagged improved/regressed/unchanged/info."""
    lines = []
    for row in rows:
        metric, before, after, delta = row["metric"], row["session_a"], row["session_b"], row["delta"]
        pct = _pct_change(before, after)
        pct_text = f"{pct:+.1f}%" if pct is not None else "n/a"

        if metric in IMPROVES_WHEN_LOWER:
            verdict = "improved" if delta < 0 else ("regressed" if delta > 0 else "unchanged")
        elif metric in IMPROVES_WHEN_HIGHER:
            verdict = "improved" if delta > 0 else ("regressed" if delta < 0 else "unchanged")
        else:
            verdict = "info"

        lines.append(f"{metric}: {before} -> {after} ({pct_text}) [{verdict}]")

    return lines


def main(db_path=None):
    tasks = find_oldest_and_newest_tasks(db_path)
    if tasks is None:
        print("Need at least 2 recorded sessions (task_ids) to compare — found fewer than that.")
        return

    oldest, newest = tasks
    rows = compute_comparison(db_path, oldest, newest)

    console = Console()
    console.print(render_comparison(rows, oldest, newest))

    print(f"\nOldest session: {oldest}\nNewest session: {newest}\n")
    for line in summarize_improvements(rows):
        print(f"  {line}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default=None)
    args = parser.parse_args()
    main(args.db_path)
