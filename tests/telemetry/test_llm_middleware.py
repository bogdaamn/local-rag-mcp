from telemetry import context
from telemetry.llm_middleware import record_llm_call


def test_cache_miss_calls_fn_and_records_a_row(tmp_path):
    db_path = tmp_path / "telemetry.db"
    context.start_task("t1")
    context.next_turn()
    calls = []

    def fn():
        calls.append(1)
        return "the answer", 10, 20

    result = record_llm_call(
        call_site="ask_llm", model="qwen3:0.6b", prompt="what is the policy",
        temperature=None, fn=fn, db_path=db_path,
    )

    assert result == "the answer"
    assert calls == [1]  # fn called exactly once on a miss

    import sqlite3
    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT task_id, turn_number, model, input_tokens, output_tokens, "
        "cached_tokens, call_site FROM llm_calls"
    ).fetchone()
    conn.close()
    assert row == ("t1", 1, "qwen3:0.6b", 10, 20, 0, "ask_llm")


def test_cache_hit_does_not_call_fn(tmp_path):
    db_path = tmp_path / "telemetry.db"
    context.start_task("t2")
    context.next_turn()

    record_llm_call(
        call_site="ask_llm", model="qwen3:0.6b", prompt="repeat question",
        temperature=None, fn=lambda: ("first answer", 10, 20), db_path=db_path,
    )

    calls = []

    def fn_should_not_run():
        calls.append(1)
        return "should not happen", 0, 0

    result = record_llm_call(
        call_site="ask_llm", model="qwen3:0.6b", prompt="repeat question",
        temperature=None, fn=fn_should_not_run, db_path=db_path,
    )

    assert result == "first answer"
    assert calls == []


def test_cache_hit_records_cached_tokens_equal_to_input_tokens(tmp_path):
    db_path = tmp_path / "telemetry.db"
    context.start_task("t3")
    context.next_turn()
    record_llm_call(
        call_site="ask_llm", model="qwen3:0.6b", prompt="q",
        temperature=None, fn=lambda: ("a", 15, 7), db_path=db_path,
    )

    record_llm_call(
        call_site="ask_llm", model="qwen3:0.6b", prompt="q",
        temperature=None, fn=lambda: ("unused", 0, 0), db_path=db_path,
    )

    import sqlite3
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute("SELECT cached_tokens FROM llm_calls ORDER BY id").fetchall()
    conn.close()
    assert rows == [(0,), (15,)]


def test_records_estimated_cost_via_pricing(tmp_path):
    db_path = tmp_path / "telemetry.db"
    context.start_task("t4")
    context.next_turn()

    record_llm_call(
        call_site="ask_llm", model="qwen3:0.6b", prompt="q",
        temperature=None, fn=lambda: ("a", 1_000_000, 1_000_000), db_path=db_path,
    )

    import sqlite3
    conn = sqlite3.connect(str(db_path))
    cost = conn.execute("SELECT estimated_cost FROM llm_calls").fetchone()[0]
    conn.close()
    assert cost == 0.40


def test_records_reasoning_tokens_as_zero(tmp_path):
    db_path = tmp_path / "telemetry.db"
    context.start_task("t5")
    context.next_turn()

    record_llm_call(
        call_site="ask_llm", model="qwen3:0.6b", prompt="q",
        temperature=None, fn=lambda: ("a", 1, 1), db_path=db_path,
    )

    import sqlite3
    conn = sqlite3.connect(str(db_path))
    reasoning = conn.execute("SELECT reasoning_tokens FROM llm_calls").fetchone()[0]
    conn.close()
    assert reasoning == 0
