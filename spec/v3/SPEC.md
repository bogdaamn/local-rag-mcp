# Agent Token-Efficiency Optimisations — Design Spec

**Status:** Approved for planning
**Date:** 2026-09-08
**Related:** [[architecture]] / [[index]] in the project's Obsidian vault (`~/Obsidian/local-rag-mcp/`)

## Purpose

Reduce token consumption, tool-output size, cost, latency, and unnecessary
LLM calls in `CompanyKBAssistant.query()`, without materially reducing answer
quality, then measure the before/after delta on a fixed 8-question benchmark
using the telemetry infrastructure from `spec/v2/SPEC.md`.

## Origin and adaptation

A generic "Agent Token-Efficiency Optimisations" spec was supplied (tool
output budgets, local CLI-first retrieval via `grep`/`find`/`sed`/`jq`,
selective sub-agent context + JSON session memory). That spec was written for
a different shape of system — a coding agent with shell access and a
sub-agent architecture. `local-rag-mcp` is a single-process assistant
(`assistant.py`) with exactly 3 MCP tools (`read_document`, `list_documents`,
`search_documents`) and no sub-agents, no shell-exposed tool, and RAG hybrid
retrieval already doing "search before full read" over the knowledge base.
The following adaptation decisions were made (via user clarification,
2026-09-08) before planning:

1. **Merge, don't stack.** This spec supersedes the previously-scoped
   "shrink `mcp_decision` context" and "cap tool output" ideas — improvements
   1 and (in spirit) 3 below absorb them.
2. **"Local CLI-based retrieval" → extend `read_document` with a `query`
   param.** No shell tool is added. Instead `read_document` gains grep-like
   in-tool filtering, since that is this project's equivalent of "search
   before reading the whole thing."
3. **"Selective sub-agent context" → dropped; "JSON session memory" →
   kept, scoped down.** There is nothing to select-and-forward since no
   sub-agents exist. The JSON session-memory idea is implemented standalone
   (decisions/changes/errors per `task_id`), independent of any sub-agent
   split, for observability during benchmarking — not for the token-savings
   reason the original spec cited.
4. **A 4th improvement is added**, absent from the original spec: replacing
   the LLM-based `_llm_decide_mcp_usage()` / `mcp_decision` call with a
   deterministic local heuristic. This directly fixes two bugs reproduced
   live during benchmarking on session `b763a55f-5ddd-47f0-abba-781f8b7a6cb6`:
   - Session `cf532ac8-...`: the decision call hallucinated a tool named
     `add_function`, which does not exist on the MCP server, on a
     "list documents" question.
   - Session `b763a55f-...`: the decision call failed to invoke the real
     `list_documents` tool on the same kind of question, and separately
     failed to make use of `read_document`'s output on a "read the full
     contents of X" question.

   It also eliminates one whole LLM call per turn, which is squarely a
   token/cost/latency reduction in the spirit of the original ask.

## The 4 improvements, as scoped for this project

### 1. Heuristic replacing the `mcp_decision` LLM call

`mcp/decision_heuristic.py::decide_tool_usage(query, contexts) -> (tool_name_or_None, args_dict_or_None)`
— same return contract as the method it replaces
(`_llm_decide_mcp_usage`), so it's a drop-in swap at the one call site in
`assistant.py::query()`. Deterministic substring/regex rules over the
lowercased user query, first match wins:

- "list"/"which"/"what" + "document(s)" → `list_documents`, `{}`.
- "search"/"find" + "document" → `search_documents`, `{"query": <topic
  extracted from after "about"/"for", else last content word>}`.
- "read" + ("full contents" or a `name.ext` token where `ext` is one of
  `txt/md/pdf/docx`) → `read_document`, `{"file_path": "docs/<name.ext>",
  "query": <topic after "about", if present, else None>}`.
- Otherwise → `(None, None)` — no tool. Matches the existing prompt's "when
  in doubt, prefer not using MCP" rule, and is the expected/interesting
  outcome (not a bug to chase) for the deliberately-ambiguous 8th benchmark
  question.

`contexts` is accepted but currently unused, kept only so the heuristic is a
type-compatible drop-in replacement for `_llm_decide_mcp_usage`.

### 2. Token-aware, budgeted `read_document` output

`read_document(file_path, query=None, max_chars=None, offset=0)` in
`mcp/server.py`. Whole-file mode (`query=None`): returns
`text[offset:offset+effective_max]` where `effective_max = max_chars or
config.READ_DOCUMENT_MAX_CHARS`. Return value is a **JSON-encoded string**
(not a bare string, unlike the other 2 tools) shaped
`{"content": str, "truncated": bool, "total_chars": int, "next_offset":
int|null}`, so truncation is machine-readable rather than embedded as prose.
Error paths (`FileNotFoundError`, access-denied, generic exception) still
return a plain (non-JSON) string, matching the other 2 tools' existing error
convention — downstream JSON-parsing falls back to using the raw string when
`json.loads` fails, so no special-casing is needed for errors.

### 3. Grep-like `query` param on `read_document` (local-retrieval adaptation)

When `query` is given: case-insensitive substring match per line (same style
as `search_documents`' existing filename matching), returns matching lines ±
2 lines of context, merging overlapping ranges. If no line matches, returns
`{"content": "No lines matching '<query>' found in <file_path>.",
"truncated": false, "total_chars": <len of whole file>, "next_offset":
null}`. If the assembled excerpt still exceeds `effective_max`, it is capped
and `truncated: true` — **pagination (`next_offset`) is scoped to whole-file
mode only**; query-mode excerpts that overflow the budget are truncated
without a continuation mechanism (deliberate scope cut — this project's docs
are 4 small files, so a search hit needing >1 page of context is not a
realistic case worth the added complexity).

Detecting truncation for telemetry: `telemetry/tool_middleware.py`
`json.loads()`s the tool result text; if it parses to a dict with
`truncated: true`, that's recorded in the new `tool_calls.truncated` column.
Non-JSON results (the other 2 tools, and error strings) parse-fail and
default to `truncated=False` — no special-casing needed.

`assistant.py` gains a small `_extract_mcp_text(mcp_result)` helper: tries
`json.loads`; if the result is a dict with a `content` key, returns
`content` (plus a short truncation note if `truncated`); otherwise returns
`mcp_result` unchanged. This is what actually gets embedded in the
`<additional_info_from_mcp_tool>` prompt block, so the LLM never sees raw
JSON envelope syntax.

### 4. Scoped-down JSON session memory

`telemetry/session_memory.py`, backed by a new `session_memory` table
(`task_id TEXT PRIMARY KEY, memory_json TEXT NOT NULL, updated_at TEXT NOT
NULL`) in `telemetry.db`. One JSON document per `task_id`, shape:

```json
{
  "session_id": "<task_id>",
  "task_id": "<task_id>",
  "status": "in_progress",
  "decisions": [{"description": "...", "reason": "...", "timestamp": "..."}],
  "changes": [{"description": "...", "timestamp": "..."}],
  "errors": [{"description": "...", "resolution": "...", "timestamp": "..."}],
  "artifacts": {"relevant_files": [], "references": []}
}
```

`record_decision(task_id, description, reason=None)`,
`record_change(task_id, description)`,
`record_error(task_id, description, resolution=None)`, and
`get_memory(task_id)` are the module's public functions — read-modify-write
against the one row per `task_id`. **Fail-open**, matching this repo's
existing telemetry convention (`llm_middleware`/`tool_middleware`): any
storage exception is caught, warned to stderr, and never propagates into
`query()`.

Wired into `assistant.py` at 3 points: after the heuristic decides a tool (or
not) → `record_decision`; when a tool call's parsed result has
`truncated: true` → `record_change`; when `_call_mcp_tool` catches an
exception → `record_error`.

**Explicitly out of scope:** sub-agent context budgets, sub-agent handoff
token limits, and anything else from the original spec's improvement #3 that
presumes a multi-agent architecture — none of that exists in this codebase.

## Benchmark

8 fixed questions (chosen to exercise all 4 improvements — pure-RAG,
explicit `list_documents`, explicit `search_documents`, explicit
`read_document` with a large result, and one deliberately ambiguous mixed-intent
question), run via `python main.py` with the questions piped over stdin from
the worktree's `src/` directory. Pre-optimization baseline already
collected: task_id `b763a55f-5ddd-47f0-abba-781f8b7a6cb6` (24 LLM calls, 8
turns, run 2026-09-08). Post-optimization run uses the identical question
text, then `python compare_sessions.py --select` produces the before/after
delta table.

## Out of scope (for this spec)

- Sub-agent architecture of any kind (none exists; not being added).
- A general-purpose shell/CLI-exposed MCP tool (`grep`/`find`/`jq` as literal
  tools) — `read_document`'s new `query` param covers the same need for this
  project's scale.
- Pagination for query-mode `read_document` excerpts (see improvement #3).
- Retention/rotation of the new `session_memory` rows.
