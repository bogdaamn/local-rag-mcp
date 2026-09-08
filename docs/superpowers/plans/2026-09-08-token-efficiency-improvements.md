# Agent Token-Efficiency Optimisations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 4 token-efficiency improvements in `local-rag-mcp`'s assistant/MCP layer — a deterministic heuristic replacing the LLM-based tool-use decision, a budgeted/truncation-aware `read_document`, a grep-like `query` param on `read_document`, and scoped-down JSON session memory — then measure the before/after delta on the same fixed 8-question benchmark already run once as the pre-optimization baseline.

**Architecture:** `mcp/decision_heuristic.py` provides a pure `decide_tool_usage(query, contexts)` function that is a drop-in, same-contract replacement for `assistant.py`'s existing `_llm_decide_mcp_usage` method, removing that method and its one LLM call entirely. `mcp/server.py::read_document` gains `query`/`max_chars`/`offset` params and switches from returning a bare string to a JSON-encoded `{"content", "truncated", "total_chars", "next_offset"}` envelope on success (error paths stay plain strings, unchanged). `telemetry/tool_middleware.py` detects `truncated` from that envelope and records it in a new `tool_calls.truncated` column (schema-migrated for pre-existing DBs). `telemetry/session_memory.py` is a new module, backed by a new `session_memory` table, holding one JSON document per `task_id` with `decisions`/`changes`/`errors` lists, updated at 3 points inside `assistant.py::query()`/`_call_mcp_tool()` — fail-open, matching this repo's existing telemetry convention.

**Tech Stack:** Python 3.10+, sqlite3 (stdlib), `re`/`json` (stdlib), pytest + monkeypatch/tmp_path (existing test stack), `rich` (existing dependency, unchanged by this plan).

**Spec:** `spec/v3/SPEC.md`

## Global Constraints

- Implementation follows TDD (test-first) per `superpowers:test-driven-development`.
- Tests mirror source structure 1:1: `src/<module>.py` → `tests/<module_path>/test_<module>.py`, matching this repo's existing convention (confirmed via `pytest.ini`'s `pythonpath = src`, so test files import modules exactly as `assistant.py`/other `src` code does — no `src.` prefix).
- Every new/modified storage-facing function accepts an optional `db_path=None` kwarg for test isolation, matching `telemetry/storage.py`'s existing pattern. The repo's `tests/conftest.py` autouse fixture already redirects the default `config.TELEMETRY_DB_PATH` per test, so most new tests can omit `db_path` and rely on that isolation — pass it explicitly only where a test needs a specific, inspectable file.
- Every new telemetry-writing code path is **fail-open**: wrap the storage call in `try/except Exception`, print a `⚠️` warning to stderr, never raise into the caller. This matches `llm_middleware.record_llm_call` and `tool_middleware.record_tool_call`'s existing behavior, and is a hard requirement per `spec/v3/SPEC.md` (a prior bug in this codebase was telemetry failing *closed*).
- No new LLM call sites are being added; one (`mcp_decision`) is being **removed**.
- `mcp/server.py`'s 3 tools remain plain Python functions decorated with `@mcp.tool`; `list_documents`/`search_documents` are unmodified by this plan — only `read_document` changes.
- Out of scope, per `spec/v3/SPEC.md`: sub-agent context budgets (no sub-agents exist), a general shell/CLI-exposed tool, pagination for query-mode `read_document` excerpts, and `session_memory` retention/rotation.

---

## File Structure

```
src/config.py                      # + READ_DOCUMENT_MAX_CHARS
src/telemetry/storage.py           # + session_memory table, tool_calls.truncated column + migration,
                                    #   insert_tool_call() gains truncated param, + get/upsert_session_memory()
src/telemetry/session_memory.py    # NEW — record_decision/record_change/record_error/get_memory
src/telemetry/tool_middleware.py   # detect truncated from JSON tool result, pass through to storage
src/mcp/server.py                  # read_document gains query/max_chars/offset + JSON envelope; + _matching_excerpt()
src/mcp/decision_heuristic.py      # NEW — decide_tool_usage(query, contexts) -> (tool_name, args) | (None, None)
src/assistant.py                   # swap _llm_decide_mcp_usage -> decide_tool_usage; + _extract_mcp_text();
                                    #   session_memory wiring in query()/_call_mcp_tool(); drop unused
                                    #   ollama/OLLAMA_MODEL/record_llm_call imports and self.llm_client

tests/test_config.py               # + READ_DOCUMENT_MAX_CHARS assertion
tests/telemetry/test_storage.py    # + session_memory table/migration/CRUD tests
tests/telemetry/test_session_memory.py   # NEW
tests/telemetry/test_tool_middleware.py  # + truncated-detection tests
tests/mcp/test_server.py           # NEW — direct read_document() tests (no test_server.py existed before)
tests/mcp/test_decision_heuristic.py     # NEW
tests/test_assistant.py            # rewritten: drop _llm_decide_mcp_usage tests, add heuristic/session-memory tests
```

---

## Task 1: Telemetry storage — `session_memory` table + `tool_calls.truncated` column

**Files:**
- Modify: `src/telemetry/storage.py`
- Test: `tests/telemetry/test_storage.py`

**Interfaces:**
- Consumes: nothing new (pure schema/storage layer).
- Produces: `storage.insert_tool_call(..., truncated=False, ...)` (new optional kwarg, backward compatible), `storage.get_session_memory(task_id, db_path=None) -> str | None`, `storage.upsert_session_memory(task_id, memory_json, timestamp=None, db_path=None) -> None`. Task 2 (`telemetry/session_memory.py`) and Task 3 (`telemetry/tool_middleware.py`) call these.

- [ ] **Step 1: Write the failing tests**

Append to `tests/telemetry/test_storage.py`:

```python
def test_get_connection_creates_session_memory_table(tmp_path):
    db_path = tmp_path / "telemetry.db"

    conn = storage.get_connection(db_path)
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    conn.close()

    assert "session_memory" in tables


def test_get_connection_adds_truncated_column_to_a_pre_existing_tool_calls_table(tmp_path):
    db_path = tmp_path / "telemetry.db"
    # Simulate a DB created before this migration: tool_calls without `truncated`.
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE tool_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            turn_number INTEGER NOT NULL,
            tool_name TEXT NOT NULL,
            input_size INTEGER NOT NULL,
            output_size INTEGER NOT NULL,
            output_tokens INTEGER NOT NULL,
            duration_ms REAL NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()

    conn = storage.get_connection(db_path)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(tool_calls)")}
    conn.close()

    assert "truncated" in columns


def test_insert_tool_call_defaults_truncated_to_false(tmp_path):
    db_path = tmp_path / "telemetry.db"

    storage.insert_tool_call(
        agent_id="a", task_id="t1", turn_number=1, tool_name="read_document",
        input_size=1, output_size=1, output_tokens=1, duration_ms=1.0, db_path=db_path,
    )

    conn = sqlite3.connect(str(db_path))
    truncated = conn.execute("SELECT truncated FROM tool_calls").fetchone()[0]
    conn.close()
    assert truncated == 0


def test_insert_tool_call_stores_truncated_true(tmp_path):
    db_path = tmp_path / "telemetry.db"

    storage.insert_tool_call(
        agent_id="a", task_id="t1", turn_number=1, tool_name="read_document",
        input_size=1, output_size=1, output_tokens=1, duration_ms=1.0,
        truncated=True, db_path=db_path,
    )

    conn = sqlite3.connect(str(db_path))
    truncated = conn.execute("SELECT truncated FROM tool_calls").fetchone()[0]
    conn.close()
    assert truncated == 1


def test_upsert_session_memory_then_get_roundtrips(tmp_path):
    db_path = tmp_path / "telemetry.db"

    storage.upsert_session_memory("t1", '{"a": 1}', db_path=db_path)

    assert storage.get_session_memory("t1", db_path=db_path) == '{"a": 1}'


def test_upsert_session_memory_overwrites_existing_row(tmp_path):
    db_path = tmp_path / "telemetry.db"

    storage.upsert_session_memory("t1", '{"a": 1}', db_path=db_path)
    storage.upsert_session_memory("t1", '{"a": 2}', db_path=db_path)

    assert storage.get_session_memory("t1", db_path=db_path) == '{"a": 2}'


def test_get_session_memory_returns_none_when_absent(tmp_path):
    db_path = tmp_path / "telemetry.db"
    assert storage.get_session_memory("missing", db_path=db_path) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run (from repo root): `venv/bin/python -m pytest tests/telemetry/test_storage.py -v`
Expected: the 7 new tests FAIL — `session_memory` table doesn't exist, `truncated` column doesn't exist, `insert_tool_call()` has no `truncated` kwarg, `get_session_memory`/`upsert_session_memory` don't exist.

- [ ] **Step 3: Implement the schema migration and new storage functions**

In `src/telemetry/storage.py`, replace `SCHEMA_SQL` with:

```python
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cached_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
    latency_ms REAL NOT NULL,
    estimated_cost REAL NOT NULL,
    call_site TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    tool_name TEXT NOT NULL,
    input_size INTEGER NOT NULL,
    output_size INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    duration_ms REAL NOT NULL,
    truncated INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    response TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS session_memory (
    task_id TEXT PRIMARY KEY,
    memory_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _ensure_tool_calls_truncated_column(conn):
    """Migration for DBs created before the `truncated` column existed —
    CREATE TABLE IF NOT EXISTS is a no-op on an already-existing table, so
    new columns need an explicit ALTER TABLE guarded by a PRAGMA check
    (SQLite has no ADD COLUMN IF NOT EXISTS)."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(tool_calls)")}
    if "truncated" not in columns:
        conn.execute("ALTER TABLE tool_calls ADD COLUMN truncated INTEGER NOT NULL DEFAULT 0")
        conn.commit()
```

Update `get_connection` to call the migration after creating the schema:

```python
def get_connection(db_path=None):
    """Open a new connection to the telemetry DB, creating the schema (and
    migrating older schemas) if needed. Callers are responsible for closing
    it."""
    resolved = Path(db_path) if db_path else _default_db_path()
    conn = sqlite3.connect(str(resolved))
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    _ensure_tool_calls_truncated_column(conn)
    return conn
```

Update `insert_tool_call` to accept and store `truncated`:

```python
def insert_tool_call(
    agent_id, task_id, turn_number, tool_name, input_size, output_size,
    output_tokens, duration_ms, truncated=False, timestamp=None, db_path=None,
):
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            INSERT INTO tool_calls (
                timestamp, agent_id, task_id, turn_number, tool_name,
                input_size, output_size, output_tokens, duration_ms, truncated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp or now_iso(), agent_id, task_id, turn_number, tool_name,
                input_size, output_size, output_tokens, duration_ms,
                int(bool(truncated)),
            ),
        )
        conn.commit()
    finally:
        conn.close()
```

Add two new functions at the end of the file:

```python
def get_session_memory(task_id, db_path=None):
    """The stored memory_json string for task_id, or None if nothing has
    been recorded yet."""
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT memory_json FROM session_memory WHERE task_id = ?", (task_id,)
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def upsert_session_memory(task_id, memory_json, timestamp=None, db_path=None):
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            INSERT INTO session_memory (task_id, memory_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                memory_json = excluded.memory_json,
                updated_at = excluded.updated_at
            """,
            (task_id, memory_json, timestamp or now_iso()),
        )
        conn.commit()
    finally:
        conn.close()
```

Add `import sqlite3` at the top of `tests/telemetry/test_storage.py` if not already present (it already is, per the existing file).

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python -m pytest tests/telemetry/test_storage.py -v`
Expected: all tests PASS (7 new + all pre-existing).

- [ ] **Step 5: Commit**

```bash
git add src/telemetry/storage.py tests/telemetry/test_storage.py
git commit -m "feat: add session_memory table and tool_calls.truncated column"
```

---

## Task 2: `telemetry/session_memory.py` module

**Files:**
- Create: `src/telemetry/session_memory.py`
- Test: `tests/telemetry/test_session_memory.py`

**Interfaces:**
- Consumes: `storage.get_session_memory(task_id, db_path=None)`, `storage.upsert_session_memory(task_id, memory_json, timestamp=None, db_path=None)`, `storage.now_iso()` (all from Task 1).
- Produces: `session_memory.get_memory(task_id, db_path=None) -> dict`, `session_memory.record_decision(task_id, description, reason=None, db_path=None, timestamp=None)`, `session_memory.record_change(task_id, description, db_path=None, timestamp=None)`, `session_memory.record_error(task_id, description, resolution=None, db_path=None, timestamp=None)`. Task 6 (`assistant.py`) calls these three `record_*` functions and `get_memory`.

- [ ] **Step 1: Write the failing tests**

Create `tests/telemetry/test_session_memory.py`:

```python
from telemetry import session_memory


def test_get_memory_returns_empty_shape_when_nothing_recorded(tmp_path):
    db_path = tmp_path / "telemetry.db"

    memory = session_memory.get_memory("t1", db_path=db_path)

    assert memory == {
        "session_id": "t1", "task_id": "t1", "status": "in_progress",
        "decisions": [], "changes": [], "errors": [],
        "artifacts": {"relevant_files": [], "references": []},
    }


def test_record_decision_appends_to_decisions_list(tmp_path):
    db_path = tmp_path / "telemetry.db"

    session_memory.record_decision("t1", "used heuristic", reason="no LLM needed", db_path=db_path)

    memory = session_memory.get_memory("t1", db_path=db_path)
    assert len(memory["decisions"]) == 1
    assert memory["decisions"][0]["description"] == "used heuristic"
    assert memory["decisions"][0]["reason"] == "no LLM needed"
    assert memory["decisions"][0]["timestamp"]


def test_record_change_appends_to_changes_list(tmp_path):
    db_path = tmp_path / "telemetry.db"

    session_memory.record_change("t1", "output truncated", db_path=db_path)

    memory = session_memory.get_memory("t1", db_path=db_path)
    assert len(memory["changes"]) == 1
    assert memory["changes"][0]["description"] == "output truncated"
    assert memory["changes"][0]["timestamp"]


def test_record_error_appends_to_errors_list(tmp_path):
    db_path = tmp_path / "telemetry.db"

    session_memory.record_error("t1", "tool call failed", resolution="retried", db_path=db_path)

    memory = session_memory.get_memory("t1", db_path=db_path)
    assert memory["errors"][0]["description"] == "tool call failed"
    assert memory["errors"][0]["resolution"] == "retried"


def test_multiple_records_accumulate_in_order(tmp_path):
    db_path = tmp_path / "telemetry.db"

    session_memory.record_decision("t1", "first", db_path=db_path)
    session_memory.record_decision("t1", "second", db_path=db_path)

    memory = session_memory.get_memory("t1", db_path=db_path)
    assert [d["description"] for d in memory["decisions"]] == ["first", "second"]


def test_different_task_ids_do_not_share_memory(tmp_path):
    db_path = tmp_path / "telemetry.db"

    session_memory.record_decision("t1", "for t1", db_path=db_path)
    session_memory.record_decision("t2", "for t2", db_path=db_path)

    assert [d["description"] for d in session_memory.get_memory("t1", db_path=db_path)["decisions"]] == ["for t1"]
    assert [d["description"] for d in session_memory.get_memory("t2", db_path=db_path)["decisions"]] == ["for t2"]


def test_record_decision_does_not_raise_when_storage_fails(tmp_path, monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr("telemetry.storage.upsert_session_memory", _boom)

    session_memory.record_decision("t1", "x", db_path=tmp_path / "telemetry.db")  # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest tests/telemetry/test_session_memory.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'telemetry.session_memory'`.

- [ ] **Step 3: Implement the module**

Create `src/telemetry/session_memory.py`:

```python
import json
import sys

from telemetry import storage
from telemetry.storage import now_iso


def _empty_memory(task_id):
    return {
        "session_id": task_id,
        "task_id": task_id,
        "status": "in_progress",
        "decisions": [],
        "changes": [],
        "errors": [],
        "artifacts": {"relevant_files": [], "references": []},
    }


def get_memory(task_id, db_path=None):
    """The parsed memory dict for task_id, or a fresh empty shape if
    nothing has been recorded yet."""
    raw = storage.get_session_memory(task_id, db_path=db_path)
    if raw is None:
        return _empty_memory(task_id)
    return json.loads(raw)


def _append(task_id, section, entry, db_path):
    try:
        memory = get_memory(task_id, db_path=db_path)
        memory[section].append(entry)
        storage.upsert_session_memory(task_id, json.dumps(memory), db_path=db_path)
    except Exception as e:
        print(f"⚠️  Failed to record session memory ({section}): {e}", file=sys.stderr)


def record_decision(task_id, description, reason=None, db_path=None, timestamp=None):
    _append(task_id, "decisions", {
        "description": description, "reason": reason,
        "timestamp": timestamp or now_iso(),
    }, db_path)


def record_change(task_id, description, db_path=None, timestamp=None):
    _append(task_id, "changes", {
        "description": description, "timestamp": timestamp or now_iso(),
    }, db_path)


def record_error(task_id, description, resolution=None, db_path=None, timestamp=None):
    _append(task_id, "errors", {
        "description": description, "resolution": resolution,
        "timestamp": timestamp or now_iso(),
    }, db_path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python -m pytest tests/telemetry/test_session_memory.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/telemetry/session_memory.py tests/telemetry/test_session_memory.py
git commit -m "feat: add telemetry.session_memory JSON session-state tracking"
```

---

## Task 3: `tool_middleware` — detect and record `truncated`

**Files:**
- Modify: `src/telemetry/tool_middleware.py`
- Test: `tests/telemetry/test_tool_middleware.py`

**Interfaces:**
- Consumes: `storage.insert_tool_call(..., truncated=False, ...)` (Task 1).
- Produces: no new public function — `record_tool_call`'s existing signature/return value is unchanged; only its telemetry side-effect gains the `truncated` column. Task 4's `read_document` JSON envelope is what makes `truncated=True` observable end-to-end once both are done.

- [ ] **Step 1: Write the failing tests**

Append to `tests/telemetry/test_tool_middleware.py` (add `import json` near the top if not already imported at module level — the existing file only imports it inline in one test):

```python
def test_records_truncated_true_when_result_json_says_so(tmp_path):
    db_path = tmp_path / "telemetry.db"
    context.start_task("t4")
    context.next_turn()

    record_tool_call(
        "read_document", {"file_path": "docs/sqlite.txt"},
        lambda: {"result": json.dumps({
            "content": "partial...", "truncated": True,
            "total_chars": 9000, "next_offset": 4000,
        })},
        db_path=db_path,
    )

    conn = sqlite3.connect(str(db_path))
    truncated = conn.execute("SELECT truncated FROM tool_calls").fetchone()[0]
    conn.close()
    assert truncated == 1


def test_records_truncated_false_when_result_json_says_so(tmp_path):
    db_path = tmp_path / "telemetry.db"
    context.start_task("t5")
    context.next_turn()

    record_tool_call(
        "read_document", {"file_path": "docs/sqlite.txt"},
        lambda: {"result": json.dumps({
            "content": "whole file", "truncated": False,
            "total_chars": 10, "next_offset": None,
        })},
        db_path=db_path,
    )

    conn = sqlite3.connect(str(db_path))
    truncated = conn.execute("SELECT truncated FROM tool_calls").fetchone()[0]
    conn.close()
    assert truncated == 0


def test_records_truncated_false_for_plain_string_results(tmp_path):
    db_path = tmp_path / "telemetry.db"
    context.start_task("t6")
    context.next_turn()

    record_tool_call(
        "list_documents", {}, lambda: {"result": "doc1\ndoc2"}, db_path=db_path,
    )

    conn = sqlite3.connect(str(db_path))
    truncated = conn.execute("SELECT truncated FROM tool_calls").fetchone()[0]
    conn.close()
    assert truncated == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest tests/telemetry/test_tool_middleware.py -v`
Expected: the 3 new tests FAIL — `sqlite3.OperationalError: no such column: truncated` (Task 1 must already be merged for the column to exist; the failure here is `record_tool_call` never passing `truncated=` through, so it's stuck at the column's default `0`, making the "truncated true" test fail its assertion).

- [ ] **Step 3: Implement truncation detection**

Replace `src/telemetry/tool_middleware.py`'s `record_tool_call` body:

```python
import json
import time

from telemetry import context, storage


def record_tool_call(tool_name, arguments, fn, db_path=None):
    """Run fn() (a zero-arg callable invoking the actual MCP tool call and
    returning its raw JSON-RPC response dict) and record one row in
    tool_calls. output_size/output_tokens are sourced from the response's
    "result" field only, not the full JSON-RPC envelope. If that result
    text itself parses as JSON with a `truncated` key (read_document's new
    envelope), that value is recorded too — plain-string results (the
    other 2 tools, and any tool's error strings) default to
    truncated=False. Returns fn()'s return value unchanged."""
    input_size = len(json.dumps(arguments).encode("utf-8"))

    start = time.perf_counter()
    response = fn()
    duration_ms = (time.perf_counter() - start) * 1000

    result = response.get("result", "") if isinstance(response, dict) else ""
    result_text = result if isinstance(result, str) else json.dumps(result)
    output_size = len(result_text.encode("utf-8"))
    output_tokens = len(result_text) // 4

    truncated = False
    try:
        parsed = json.loads(result_text)
        if isinstance(parsed, dict):
            truncated = bool(parsed.get("truncated", False))
    except (json.JSONDecodeError, TypeError):
        pass

    try:
        storage.insert_tool_call(
            agent_id=context.current_agent_id.get(),
            task_id=context.current_task_id.get() or "unattributed",
            turn_number=context.current_turn_number.get(),
            tool_name=tool_name,
            input_size=input_size,
            output_size=output_size,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
            truncated=truncated,
            db_path=db_path,
        )
    except Exception as e:
        import sys
        print(f"⚠️  Failed to record tool call telemetry: {e}", file=sys.stderr)

    return response
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python -m pytest tests/telemetry/test_tool_middleware.py -v`
Expected: all tests PASS (3 new + all pre-existing).

- [ ] **Step 5: Commit**

```bash
git add src/telemetry/tool_middleware.py tests/telemetry/test_tool_middleware.py
git commit -m "feat: record tool-output truncation in tool_calls telemetry"
```

---

## Task 4: Budgeted, grep-capable `read_document`

**Files:**
- Modify: `src/config.py`
- Modify: `src/mcp/server.py`
- Test: `tests/test_config.py`
- Test: `tests/mcp/test_server.py` (new file)

**Interfaces:**
- Consumes: `config.DOCUMENTS_DIR` (existing), `config.READ_DOCUMENT_MAX_CHARS` (new, this task).
- Produces: `read_document(file_path, query=None, max_chars=None, offset=0) -> str` — on success, a JSON-encoded string `{"content": str, "truncated": bool, "total_chars": int, "next_offset": int|None}`; on error, a plain (non-JSON) string, unchanged from before. Task 6 (`assistant.py::_extract_mcp_text`) and Task 3 (`tool_middleware`, already done) both parse this shape.

- [ ] **Step 1: Write the failing tests**

Add one line to `tests/test_config.py`'s `test_telemetry_config_constants` — actually add a new, separate test function so it's not conflated with telemetry constants:

```python
def test_read_document_max_chars_default():
    assert config.READ_DOCUMENT_MAX_CHARS == 4000
```

Create `tests/mcp/test_server.py`:

```python
import json

import pytest

from mcp.server import read_document, list_documents, search_documents


@pytest.fixture
def docs_dir(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "sample.txt").write_text(
        "\n".join(f"line {i}" for i in range(1, 11)).replace("line 5", "line 5 mentions avgdl")
    )
    monkeypatch.setattr("mcp.server.DOCUMENTS_DIR", str(docs))
    return docs


def test_read_document_whole_file_under_budget_is_not_truncated(docs_dir):
    result = json.loads(read_document(str(docs_dir / "sample.txt")))
    assert result["truncated"] is False
    assert result["next_offset"] is None
    assert "line 1" in result["content"]
    assert "mentions avgdl" in result["content"]


def test_read_document_whole_file_over_budget_is_truncated_with_next_offset(docs_dir):
    result = json.loads(read_document(str(docs_dir / "sample.txt"), max_chars=10))
    assert result["truncated"] is True
    assert result["next_offset"] == 10
    assert len(result["content"]) == 10


def test_read_document_offset_continues_from_next_offset(docs_dir):
    first = json.loads(read_document(str(docs_dir / "sample.txt"), max_chars=10))
    second = json.loads(
        read_document(str(docs_dir / "sample.txt"), max_chars=10, offset=first["next_offset"])
    )
    full_text = (docs_dir / "sample.txt").read_text()
    assert first["content"] + second["content"] == full_text[:20]


def test_read_document_with_query_returns_only_matching_lines_with_context(docs_dir):
    result = json.loads(read_document(str(docs_dir / "sample.txt"), query="avgdl"))
    assert "mentions avgdl" in result["content"]
    assert "line 9" not in result["content"]
    assert result["truncated"] is False


def test_read_document_with_query_no_match_reports_zero_hits(docs_dir):
    result = json.loads(read_document(str(docs_dir / "sample.txt"), query="nonexistent-term"))
    assert "No lines matching" in result["content"]
    assert result["truncated"] is False


def test_read_document_with_query_merges_nearby_matches_into_one_range(tmp_path, monkeypatch):
    docs = tmp_path / "docs2"
    docs.mkdir()
    (docs / "merge.txt").write_text("a\nb avgdl\nc\nd avgdl\ne\nf\n")
    monkeypatch.setattr("mcp.server.DOCUMENTS_DIR", str(docs))

    result = json.loads(read_document(str(docs / "merge.txt"), query="avgdl"))

    assert "\n...\n" not in result["content"]
    assert result["content"].count("avgdl") == 2


def test_read_document_file_not_found_returns_plain_error_string(docs_dir):
    missing = docs_dir / "missing.txt"
    result = read_document(str(missing))
    assert result == f"Error: File not found: {missing}"


def test_read_document_access_denied_outside_documents_dir(docs_dir, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    result = read_document(str(outside))
    assert result.startswith("Error: Access denied")


def test_list_documents_and_search_documents_are_unaffected(docs_dir):
    assert "sample.txt" in list_documents()
    assert "sample.txt" in search_documents("sample")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_config.py tests/mcp/test_server.py -v`
Expected: `test_read_document_max_chars_default` FAILs with `AttributeError`; every `test_read_document_*` test FAILs (either a `TypeError` on unexpected kwargs, or `json.loads` raising `JSONDecodeError` because `read_document` still returns a bare string).

- [ ] **Step 3: Implement**

In `src/config.py`, add after `TELEMETRY_DB_PATH`/`AGENT_ID`:

```python
# Token-efficiency: default output budget for read_document (chars, not tokens —
# no tokenizer is available for arbitrary document text; see spec/v3/SPEC.md)
READ_DOCUMENT_MAX_CHARS = 4000
```

Replace `src/mcp/server.py` in full:

```python
import json
from fastmcp import FastMCP
from pathlib import Path
import sys

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DOCUMENTS_DIR, READ_DOCUMENT_MAX_CHARS

mcp = FastMCP("doc-tools", version="1.0.0")


def _matching_excerpt(text, query, context_lines=2):
    """Grep-like: lines containing `query` (case-insensitive substring),
    each surrounded by `context_lines` lines of context; overlapping
    ranges are merged, non-adjacent ranges are joined with a separator.
    Returns None if `query` matches no line."""
    lines = text.splitlines()
    query_lower = query.lower()
    matches = [i for i, line in enumerate(lines) if query_lower in line.lower()]

    if not matches:
        return None

    ranges = []
    for i in matches:
        start = max(0, i - context_lines)
        end = min(len(lines), i + context_lines + 1)
        if ranges and start <= ranges[-1][1]:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
        else:
            ranges.append((start, end))

    parts = ["\n".join(lines[start:end]) for start, end in ranges]
    return "\n...\n".join(parts)


@mcp.tool
def read_document(file_path: str, query: str = None, max_chars: int = None, offset: int = 0) -> str:
    """Reads a document from the knowledge base. Without `query`, returns a
    budgeted window of the whole file (text[offset:offset+max_chars],
    default budget READ_DOCUMENT_MAX_CHARS), reporting truncation and a
    next_offset for continuation. With `query`, returns only the lines
    matching it (case-insensitive substring) plus surrounding context,
    instead of the whole file. On success, returns a JSON-encoded string
    {"content", "truncated", "total_chars", "next_offset"}. On error,
    returns a plain (non-JSON) string, same as the other 2 tools."""
    try:
        path = Path(file_path)
        # Security: ensure path is within documents directory
        if not str(path.resolve()).startswith(str(Path(DOCUMENTS_DIR).resolve())):
            return f"Error: Access denied. File must be in {DOCUMENTS_DIR}"

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        effective_max = max_chars or READ_DOCUMENT_MAX_CHARS

        if query:
            excerpt = _matching_excerpt(text, query)
            if excerpt is None:
                return json.dumps({
                    "content": f"No lines matching '{query}' found in {file_path}.",
                    "truncated": False, "total_chars": len(text), "next_offset": None,
                })
            truncated = len(excerpt) > effective_max
            return json.dumps({
                "content": excerpt[:effective_max],
                "truncated": truncated, "total_chars": len(excerpt), "next_offset": None,
            })

        window = text[offset:offset + effective_max]
        truncated = offset + len(window) < len(text)
        return json.dumps({
            "content": window,
            "truncated": truncated,
            "total_chars": len(text),
            "next_offset": offset + effective_max if truncated else None,
        })
    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except Exception as e:
        return f"Error reading file: {str(e)}"


@mcp.tool
def list_documents() -> str:
    """Lists all available documents in the knowledge base."""
    try:
        base_dir = Path(DOCUMENTS_DIR)
        if not base_dir.exists():
            return f"Error: Documents directory {DOCUMENTS_DIR} does not exist"

        documents = []
        for path in base_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".txt", ".md", ".pdf", ".docx"}:
                documents.append(str(path.relative_to(base_dir)))

        if not documents:
            return "No documents found in the knowledge base."

        return "\n".join(f"- {doc}" for doc in sorted(documents))
    except Exception as e:
        return f"Error listing documents: {str(e)}"


@mcp.tool
def search_documents(query: str) -> str:
    """Searches for documents by name (case-insensitive)."""
    try:
        base_dir = Path(DOCUMENTS_DIR)
        if not base_dir.exists():
            return f"Error: Documents directory {DOCUMENTS_DIR} does not exist"

        query_lower = query.lower()
        matches = []

        for path in base_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".txt", ".md", ".pdf", ".docx"}:
                if query_lower in path.name.lower():
                    matches.append(str(path.relative_to(base_dir)))

        if not matches:
            return f"No documents found matching '{query}'"

        return "\n".join(f"- {doc}" for doc in sorted(matches))
    except Exception as e:
        return f"Error searching documents: {str(e)}"


if __name__ == "__main__":
    # Run MCP server (stdio)
    mcp.run()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_config.py tests/mcp/test_server.py -v`
Expected: all tests PASS (10 new total: 1 config + 9 server).

- [ ] **Step 5: Commit**

```bash
git add src/config.py src/mcp/server.py tests/test_config.py tests/mcp/test_server.py
git commit -m "feat: budget and grep-filter read_document's output"
```

---

## Task 5: `mcp/decision_heuristic.py` — deterministic tool-use decision

**Files:**
- Create: `src/mcp/decision_heuristic.py`
- Test: `tests/mcp/test_decision_heuristic.py`

**Interfaces:**
- Consumes: nothing (pure function, no imports beyond `re`).
- Produces: `decide_tool_usage(query, contexts) -> (str, dict) | (None, None)`. Task 6 (`assistant.py`) imports and calls this in place of `_llm_decide_mcp_usage`.

- [ ] **Step 1: Write the failing tests**

Create `tests/mcp/test_decision_heuristic.py`:

```python
from mcp.decision_heuristic import decide_tool_usage


def test_list_documents_question_triggers_list_documents_with_no_args():
    tool, args = decide_tool_usage("List all the documents in the knowledge base.", [])
    assert (tool, args) == ("list_documents", {})


def test_which_documents_question_also_triggers_list_documents():
    tool, args = decide_tool_usage("Which documents exist about ranking?", [])
    assert tool == "list_documents"


def test_search_command_triggers_search_documents_with_extracted_topic():
    tool, args = decide_tool_usage("Search the knowledge base for anything about embeddings.", [])
    assert tool == "search_documents"
    assert args == {"query": "embeddings"}


def test_find_command_triggers_search_documents_with_extracted_topic():
    tool, args = decide_tool_usage("Find the document about sentence embeddings", [])
    assert tool == "search_documents"
    assert args == {"query": "sentence embeddings"}


def test_read_full_contents_of_named_file_triggers_read_document():
    tool, args = decide_tool_usage("Read the full contents of sqlite.txt.", [])
    assert tool == "read_document"
    assert args == {"file_path": "docs/sqlite.txt"}


def test_read_document_extracts_topic_when_about_is_present():
    tool, args = decide_tool_usage("Read the full contents of sqlite.txt about WAL mode.", [])
    assert tool == "read_document"
    assert args == {"file_path": "docs/sqlite.txt", "query": "WAL mode"}


def test_read_without_a_recognizable_filename_uses_no_tool():
    tool, args = decide_tool_usage("Can you read me the contents of the policy?", [])
    assert (tool, args) == (None, None)


def test_pure_rag_question_with_no_document_keyword_uses_no_tool():
    tool, args = decide_tool_usage("What is BM25 and how does it use avgdl in ranking?", [])
    assert (tool, args) == (None, None)


def test_search_word_in_a_conceptual_question_does_not_trigger_a_tool():
    tool, args = decide_tool_usage(
        "What is FTS5 in SQLite and how does it relate to full-text search?", [],
    )
    assert (tool, args) == (None, None)


def test_cross_document_synthesis_question_uses_no_tool():
    tool, args = decide_tool_usage(
        "How do sentence embeddings differ from BM25 ranking for search relevance?", [],
    )
    assert (tool, args) == (None, None)


def test_mixed_intent_question_resolves_to_list_documents():
    tool, args = decide_tool_usage(
        "What was Robertson's contribution to BM25, and what other documents "
        "exist about ranking or search?", [],
    )
    assert tool == "list_documents"


def test_contexts_argument_is_accepted_but_does_not_affect_the_decision():
    tool_a, args_a = decide_tool_usage("avgdl", [])
    tool_b, args_b = decide_tool_usage("avgdl", [{"source": "x", "text": "y"}])
    assert (tool_a, args_a) == (tool_b, args_b) == (None, None)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest tests/mcp/test_decision_heuristic.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mcp.decision_heuristic'`.

- [ ] **Step 3: Implement**

Create `src/mcp/decision_heuristic.py`:

```python
import re

# Matches config.DOCUMENTS_DIR = "./docs", without the leading "./" — the
# assembled file_path is what gets passed to read_document's own path
# resolution, so it must match that convention.
DOCUMENTS_PATH_PREFIX = "docs"

FILENAME_RE = re.compile(r"\b([\w\-]+\.(?:txt|md|pdf|docx))\b", re.IGNORECASE)


def _extract_topic(query):
    """Best-effort topic extraction for search_documents'/read_document's
    `query` arg: text after the last ' about ' or ' for ', else the
    query's last word. Approximate on purpose — deterministic heuristic,
    not an LLM."""
    lower = query.lower()
    for marker in (" about ", " for "):
        if marker in lower:
            idx = lower.rindex(marker) + len(marker)
            topic = query[idx:].strip().rstrip("?.!")
            if topic:
                return topic
    words = query.strip().rstrip("?.!").split()
    return words[-1] if words else query


def decide_tool_usage(query, contexts):
    """Deterministic, rule-based replacement for the old LLM-based
    _llm_decide_mcp_usage/mcp_decision call. Same return contract as the
    method it replaces: (tool_name, args) if a tool should be used, else
    (None, None). `contexts` is accepted but unused — kept only so this is
    a drop-in-compatible signature for assistant.py::query()'s one call
    site."""
    lower = query.strip().lower()

    if lower.startswith(("list", "which", "what")) and "document" in lower:
        return "list_documents", {}

    if lower.startswith(("search", "find")):
        return "search_documents", {"query": _extract_topic(query)}

    filename_match = FILENAME_RE.search(query)
    if "read" in lower and ("full contents" in lower or "contents of" in lower or filename_match):
        if not filename_match:
            return None, None
        args = {"file_path": f"{DOCUMENTS_PATH_PREFIX}/{filename_match.group(1)}"}
        if " about " in lower:
            args["query"] = _extract_topic(query)
        return "read_document", args

    return None, None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python -m pytest tests/mcp/test_decision_heuristic.py -v`
Expected: all 12 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mcp/decision_heuristic.py tests/mcp/test_decision_heuristic.py
git commit -m "feat: add deterministic heuristic to replace the mcp_decision LLM call"
```

---

## Task 6: Wire the heuristic + session memory into `assistant.py`

**Files:**
- Modify: `src/assistant.py`
- Modify: `tests/test_assistant.py`

**Interfaces:**
- Consumes: `mcp.decision_heuristic.decide_tool_usage` (Task 5), `telemetry.session_memory.record_decision/record_change/record_error/get_memory` (Task 2), the JSON envelope shape produced by `read_document` (Task 4).
- Produces: `CompanyKBAssistant._extract_mcp_text(mcp_result) -> str` (new method). `CompanyKBAssistant.query()`'s public return shape (`{"answer", "sources", "mcp_used", "mcp_tool"}`) is unchanged. `_llm_decide_mcp_usage` is removed entirely — nothing downstream of this task may reference it.

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_assistant.py` in full:

```python
import json

import assistant as assistant_module
from assistant import CompanyKBAssistant
from telemetry import context, session_memory


class _FakeMCPClient:
    def __init__(self, *args, **kwargs):
        pass

    def close(self):
        pass


def _make_bare_assistant(mcp=None, task_id="test-task"):
    instance = CompanyKBAssistant.__new__(CompanyKBAssistant)
    instance.mcp = mcp
    instance.task_id = task_id
    return instance


def test_init_starts_a_task_in_ambient_context(monkeypatch):
    monkeypatch.setattr(assistant_module, "MCPClient", _FakeMCPClient)

    instance = CompanyKBAssistant()

    assert instance.task_id == context.current_task_id.get()
    assert context.current_turn_number.get() == 0


def test_query_advances_turn_number_once_per_call(monkeypatch):
    monkeypatch.setattr(assistant_module, "MCPClient", _FakeMCPClient)
    instance = CompanyKBAssistant()

    monkeypatch.setattr(assistant_module, "retrieve", lambda q: [])
    monkeypatch.setattr(assistant_module, "build_prompt", lambda q, c: "prompt")
    monkeypatch.setattr(assistant_module, "ask_llm", lambda prompt: "answer")
    monkeypatch.setattr(assistant_module, "decide_tool_usage", lambda q, c: (None, None))

    instance.query("first question")
    assert context.current_turn_number.get() == 1

    instance.query("second question")
    assert context.current_turn_number.get() == 2


def test_query_delegates_tool_decision_to_the_heuristic_not_an_llm_call(monkeypatch):
    monkeypatch.setattr(assistant_module, "MCPClient", _FakeMCPClient)
    instance = CompanyKBAssistant()

    captured = {}
    monkeypatch.setattr(assistant_module, "retrieve", lambda q: [])
    monkeypatch.setattr(assistant_module, "build_prompt", lambda q, c: "prompt")
    monkeypatch.setattr(assistant_module, "ask_llm", lambda prompt: "answer")

    def fake_decide(query, contexts):
        captured["query"] = query
        captured["contexts"] = contexts
        return None, None

    monkeypatch.setattr(assistant_module, "decide_tool_usage", fake_decide)

    instance.query("list all documents")

    assert captured["query"] == "list all documents"
    assert captured["contexts"] == []


def test_query_records_a_decision_in_session_memory(monkeypatch):
    monkeypatch.setattr(assistant_module, "MCPClient", _FakeMCPClient)
    instance = CompanyKBAssistant()

    monkeypatch.setattr(assistant_module, "retrieve", lambda q: [])
    monkeypatch.setattr(assistant_module, "build_prompt", lambda q, c: "prompt")
    monkeypatch.setattr(assistant_module, "ask_llm", lambda prompt: "answer")
    monkeypatch.setattr(assistant_module, "decide_tool_usage", lambda q, c: (None, None))

    instance.query("a pure RAG question")

    memory = session_memory.get_memory(instance.task_id)
    assert len(memory["decisions"]) == 1
    assert "no tool" in memory["decisions"][0]["description"].lower()


def test_extract_mcp_text_returns_plain_string_unchanged():
    instance = _make_bare_assistant()
    assert instance._extract_mcp_text("doc1\ndoc2") == "doc1\ndoc2"


def test_extract_mcp_text_unwraps_json_content_envelope():
    instance = _make_bare_assistant()
    payload = json.dumps({
        "content": "the file text", "truncated": False,
        "total_chars": 13, "next_offset": None,
    })
    assert instance._extract_mcp_text(payload) == "the file text"


def test_extract_mcp_text_appends_truncation_note_when_truncated():
    instance = _make_bare_assistant()
    payload = json.dumps({
        "content": "partial text", "truncated": True,
        "total_chars": 9000, "next_offset": 4000,
    })
    result = instance._extract_mcp_text(payload)
    assert result.startswith("partial text")
    assert "truncated" in result.lower()


def test_call_mcp_tool_records_an_error_in_session_memory_on_exception():
    class _BoomMCP:
        def call_tool(self, tool_name, tool_args):
            raise ConnectionError("mcp server gone")

    instance = _make_bare_assistant(mcp=_BoomMCP())

    instance._call_mcp_tool("read_document", {"file_path": "docs/x.txt"})

    memory = session_memory.get_memory(instance.task_id)
    assert len(memory["errors"]) == 1
    assert "read_document" in memory["errors"][0]["description"]


def test_query_records_a_change_when_tool_output_is_truncated(monkeypatch):
    monkeypatch.setattr(assistant_module, "MCPClient", _FakeMCPClient)
    instance = CompanyKBAssistant()

    monkeypatch.setattr(assistant_module, "retrieve", lambda q: [])
    monkeypatch.setattr(assistant_module, "build_prompt", lambda q, c: "prompt")
    monkeypatch.setattr(assistant_module, "ask_llm", lambda prompt: "answer")
    monkeypatch.setattr(
        assistant_module, "decide_tool_usage",
        lambda q, c: ("read_document", {"file_path": "docs/sqlite.txt"}),
    )
    monkeypatch.setattr(
        instance, "_call_mcp_tool",
        lambda tool_name, tool_args: json.dumps({
            "content": "partial", "truncated": True,
            "total_chars": 9000, "next_offset": 4000,
        }),
    )

    instance.query("read the full contents of sqlite.txt")

    memory = session_memory.get_memory(instance.task_id)
    assert len(memory["changes"]) == 1
    assert "truncated" in memory["changes"][0]["description"].lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_assistant.py -v`
Expected: `test_extract_mcp_text_*` FAIL with `AttributeError: 'CompanyKBAssistant' object has no attribute '_extract_mcp_text'`; the session-memory tests FAIL (no wiring yet); `test_query_delegates_tool_decision_to_the_heuristic_not_an_llm_call` FAILs because `assistant_module` has no `decide_tool_usage` name to monkeypatch yet (`AttributeError`).

- [ ] **Step 3: Implement**

Replace `src/assistant.py` in full:

```python
import json
import sys
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from rag.query import retrieve, build_prompt, ask_llm
from mcp.client import MCPClient
from mcp.decision_heuristic import decide_tool_usage
from telemetry import context, session_memory

class CompanyKBAssistant:
    """Company Knowledge Base Assistant combining RAG and MCP."""

    def __init__(self):
        self.mcp = None
        self.task_id = context.start_task()
        self._init_mcp()

    def _init_mcp(self):
        """Initialize MCP client."""
        try:
            import sys
            from pathlib import Path
            python_cmd = sys.executable
            # Get absolute path to MCP server
            mcp_path = Path(__file__).parent / "mcp" / "server.py"
            self.mcp = MCPClient([python_cmd, str(mcp_path)])
        except Exception as e:
            print(f"Warning: Could not initialize MCP client: {e}")
            self.mcp = None

    def _extract_mcp_text(self, mcp_result):
        """MCP tool results are usually plain strings, but read_document
        returns a JSON-encoded {"content", "truncated", ...} envelope on
        success (its error strings, and both other tools' results, stay
        plain). Returns the text that should actually be embedded in the
        prompt — never raw JSON envelope syntax."""
        if not mcp_result:
            return mcp_result
        try:
            parsed = json.loads(mcp_result)
        except (json.JSONDecodeError, TypeError):
            return mcp_result
        if isinstance(parsed, dict) and "content" in parsed:
            text = parsed["content"]
            if parsed.get("truncated"):
                text += "\n\n[Note: this document was truncated to fit the output budget.]"
            return text
        return mcp_result

    def _call_mcp_tool(self, tool_name: str, tool_args: dict):
        """Call an MCP tool with given name and arguments."""
        if not self.mcp:
            return None

        try:
            result = self.mcp.call_tool(tool_name, tool_args)
            return result.get("result", "")
        except Exception as e:
            session_memory.record_error(
                self.task_id, f"MCP tool call '{tool_name}' raised an exception",
                resolution=str(e),
            )
            return f"Error calling MCP tool {tool_name}: {str(e)}"

    def query(self, user_query: str, verbose=False):
        """Answer a question using RAG and optionally MCP tools."""
        context.next_turn()
        # Step 1: Retrieve from RAG
        contexts = retrieve(user_query)

        if verbose:
            print(f"📚 Retrieved {len(contexts)} relevant chunks from knowledge base")

        # Step 2: Decide (heuristically — no LLM call) if an MCP tool is needed
        mcp_result = None
        mcp_tool_used = None
        tool_name, tool_args = decide_tool_usage(user_query, contexts)
        session_memory.record_decision(
            self.task_id,
            f"Heuristic selected tool '{tool_name}'" if tool_name else "Heuristic selected no tool",
            reason=f"query={user_query!r}",
        )

        if tool_name:
            if verbose:
                print(f"🔧 Heuristic decided to use MCP tool: {tool_name} with args: {tool_args}")
            mcp_result = self._call_mcp_tool(tool_name, tool_args)
            mcp_tool_used = tool_name
            if verbose and mcp_result:
                print(f"✅ MCP tool returned result (length: {len(mcp_result)} chars)")

        # Step 3: Build prompt with RAG context
        prompt = build_prompt(user_query, contexts)

        # Step 4: Add MCP result if available
        if mcp_result:
            mcp_text = self._extract_mcp_text(mcp_result)
            prompt += f"\n\n<additional_info_from_mcp_tool>\n{mcp_text}\n</additional_info_from_mcp_tool>\n"

            try:
                parsed = json.loads(mcp_result)
            except (json.JSONDecodeError, TypeError):
                parsed = None
            if isinstance(parsed, dict) and parsed.get("truncated"):
                session_memory.record_change(
                    self.task_id, f"Tool '{mcp_tool_used}' output was truncated",
                )

        # Step 5: Generate answer
        answer = ask_llm(prompt)

        # Step 6: Prepare response with sources
        sources = [c["source"] for c in contexts] if contexts else []

        return {
            "answer": answer,
            "sources": sources,
            "mcp_used": mcp_result is not None,
            "mcp_tool": mcp_tool_used
        }

    def close(self):
        """Clean up resources."""
        if self.mcp:
            self.mcp.close()


if __name__ == "__main__":
    assistant = CompanyKBAssistant()

    print("🤖 Company Knowledge Base Assistant")
    print("Type 'exit' or 'quit' to stop\n")

    try:
        while True:
            query = input("❓ Question: ")
            if query.lower() in {"exit", "quit"}:
                break

            print("\n" + "─" * 60)
            result = assistant.query(query, verbose=True)

            print("\n🤖 Answer:\n")

            console = Console(force_terminal=True)
            console.print(Markdown(result["answer"]))

            if result["sources"]:
                print("\n📚 Sources:")
                seen_sources = set()
                for src in result["sources"]:
                    if src not in seen_sources:
                        print(f"  • {src}")
                        seen_sources.add(src)

            if result["mcp_used"]:
                print(f"\n🔧 Used MCP tool: {result['mcp_tool']}")

            print("─" * 60 + "\n")

    finally:
        assistant.close()
```

Note what was deliberately removed vs. the pre-Task-6 version: `import ollama`, `from config import OLLAMA_MODEL`, `from telemetry.llm_middleware import record_llm_call`, `self.llm_client = ollama.Client()`, and the entire `_llm_decide_mcp_usage` method — all dead code once the heuristic replaces the LLM-based decision. `import json` is *kept* (still needed, now by `_extract_mcp_text` and step 4's truncation check, not by the removed method).

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_assistant.py -v`
Expected: all 9 tests PASS.

- [ ] **Step 5: Run the full test suite**

Run (from repo root): `venv/bin/python -m pytest -q`
Expected: all tests PASS — this is the first point where every prior task's code is exercised together (`mcp_decision`'s removal, `read_document`'s new shape, and the session-memory wiring all interact through `assistant.py`).

- [ ] **Step 6: Commit**

```bash
git add src/assistant.py tests/test_assistant.py
git commit -m "feat: replace mcp_decision LLM call with heuristic; wire session memory"
```

---

## Task 7: Benchmark verification — rerun the fixed 8 questions, compare before/after

This task has no source changes — it's the plan's payoff step, confirming the 4 improvements actually move the numbers on the same benchmark already used for the pre-optimization baseline (`b763a55f-5ddd-47f0-abba-781f8b7a6cb6`, 24 LLM calls / 8 turns / 35,953 tokens / $0.0072, collected 2026-09-08).

- [ ] **Step 1: Confirm the full test suite passes and Ollama is reachable**

From repo root:

```bash
venv/bin/python -m pytest -q
curl -s -o /dev/null -w "ollama http status: %{http_code}\n" -m 3 http://localhost:11434
```

Expected: all tests pass; Ollama returns `200`. If Ollama isn't running: `nohup ollama serve > /tmp/ollama_rerun.log 2>&1 & disown`, wait ~3s, re-check.

- [ ] **Step 2: Snapshot telemetry.db sessions before the run**

From `src/`:

```bash
sqlite3 telemetry.db "SELECT task_id, COUNT(*), MAX(timestamp) FROM llm_calls GROUP BY task_id;"
```

Record the current set of `task_id`s so the new one is unambiguous afterward.

- [ ] **Step 3: Run the identical 8 questions from the worktree's `src/` directory**

This MUST run from `.claude/worktrees/llm-tool-telemetry/src` (not the main checkout — a previously-hit bug: the `start local-rag` alias points at the main checkout, which lacks this branch's code entirely).

```bash
cd .claude/worktrees/llm-tool-telemetry/src
../venv/bin/python main.py <<'EOF'
What is BM25 and how does it use avgdl in ranking?
What is FTS5 in SQLite and how does it relate to full-text search?
What does ROWID mean in SQLite, and how does WAL mode relate to it?
List all the documents in the knowledge base.
Search the knowledge base for anything about embeddings.
Read the full contents of sqlite.txt.
How do sentence embeddings differ from BM25 ranking for search relevance?
What was Robertson's contribution to BM25, and what other documents exist about ranking or search?
exit
EOF
```

Watch for: no Python tracebacks; question 4 should show **no** `🔧` line at all if the heuristic decides no tool is warranted, or a correct `🔧 Heuristic decided to use MCP tool: list_documents` line — either is a valid heuristic outcome, but a hallucinated tool name (anything other than `read_document`/`list_documents`/`search_documents`) would indicate a bug, since the heuristic can no longer invent tool names the way the old LLM call did.

- [ ] **Step 4: Identify the new session and inspect its session memory**

```bash
sqlite3 telemetry.db "SELECT task_id, COUNT(*), MAX(timestamp) FROM llm_calls GROUP BY task_id ORDER BY MAX(timestamp);"
```

The newest `task_id` should show **16** LLM calls, not 24 — 8 turns × 2 calls (`query_expansion` + `ask_llm`) instead of 3 (`mcp_decision` + `query_expansion` + `ask_llm`), since `mcp_decision` no longer exists as a call site. Then:

```bash
../venv/bin/python -c "
from telemetry import session_memory
import json
print(json.dumps(session_memory.get_memory('<the new task_id>'), indent=2))
"
```

Expected: 8 entries in `decisions` (one per turn), and a `changes` entry if question 6 ("read the full contents of sqlite.txt") triggered truncation (likely, since `sqlite.txt` is ~13K chars against a 4000-char default budget).

- [ ] **Step 5: Compare old vs. new**

```bash
../venv/bin/python compare_sessions.py --select
```

Select the pre-optimization baseline (`b763a55f-5ddd-47f0-abba-781f8b7a6cb6`) as session A and the new task_id as session B. Confirm: **Total LLM calls** drops (24 → 16), **input/output tokens** and **estimated cost** drop, and the **per-call-site breakdown** for session B no longer lists `mcp_decision` at all.

- [ ] **Step 6: Record the result**

No commit needed for this task (no source changed) — report the before/after numbers back in conversation so they can be written up. If the numbers *don't* show an improvement (e.g., a bug makes the heuristic over-trigger tool calls, adding tool-call overhead that offsets the removed LLM call), treat that as a real finding to debug, not something to paper over before reporting.
