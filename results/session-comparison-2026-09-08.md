# Session Comparison Report

**Generated:** 2026-09-08 14:49 CEST
**Command run:** `venv/bin/python src/compare_sessions.py` (from the worktree root)

Default mode compares the oldest and newest recorded sessions in `telemetry.db`:

- **Session A (oldest):** `cf532ac8-d8bb-445c-898c-060d42dfb63b` (started 2026-09-07T12:28:55.881785+00:00, 8 turns)
- **Session B (newest):** `4b7246ea-d63b-4a1a-9837-6a431e96c924` (started 2026-09-08T11:45:51.532491+00:00, 8 turns)

## Comparison

| Metric | Session A (cf532ac8) | Session B (4b7246ea) | Δ | Δ % |
|---|---|---|---|---|
| Input tokens | 35255 | 26728 | -8527 | -24.2% |
| Output tokens | 6648 | 5213 | -1435 | -21.6% |
| Cached tokens | 0 | 17459 | +17459 | n/a |
| Estimated cost | $0.0084 | $0.0064 | -$0.0020 | -23.8% |
| Turns | 8.0 | 8.0 | 0.0 | +0.0% |
| Tool calls | 2.0 | 4.0 | +2.0 | +100.0% |
| Total LLM calls | 24 | 16 | -8 | -33.3% |
| Cache hit rate | 0.0% | 81.2% | +81.2% | n/a |

### LLM calls by call site

**Session A (cf532ac8):**

| Call site | Calls | Input tokens | Output tokens | Cost |
|---|---|---|---|---|
| ask_llm | 8 | 29997 | 2289 | $0.0065 |
| mcp_decision | 8 | 4895 | 2754 | $0.0015 |
| query_expansion | 8 | 363 | 1605 | $0.0004 |

**Session B (4b7246ea):**

| Call site | Calls | Input tokens | Output tokens | Cost |
|---|---|---|---|---|
| ask_llm | 8 | 26307 | 3302 | $0.0059 |
| query_expansion | 8 | 421 | 1911 | $0.0005 |

Note `mcp_decision` is entirely absent from Session B's breakdown — that call site was removed (replaced by a deterministic heuristic).

### Tool usage

**Session A (cf532ac8):**

| Tool | Calls | Output size (B) | Output tokens |
|---|---|---|---|
| read_document | 1 | 289 | 72 |
| add_function | 1 | 88 | 22 |

**Session B (4b7246ea):**

| Tool | Calls | Output size (B) | Output tokens |
|---|---|---|---|
| list_documents | 2 | 160 | 40 |
| search_documents | 1 | 25 | 6 |
| read_document | 1 | 4123 | 1030 |

Note Session A's tool usage includes a call to `add_function` — a tool that does not exist on the MCP server (`read_document`/`list_documents`/`search_documents` are the only 3 real tools); this was a hallucinated tool call from the old LLM-based decision step.

## Regressed metrics — takeaways

- **Tool calls (+100.0%, 2 → 4):** This reads as a regression under a "lower is better" convention, but it is not a real regression here — Session A's LLM-based tool-use decision unreliably invoked tools (only 1 legitimate call out of what should have been 3 tool-worthy questions, plus 1 hallucinated call to a nonexistent tool), while Session B's deterministic heuristic correctly invokes a real tool on every question that needs one. More tool calls in this case means *more correct* tool usage, not wasted work.

All other metrics improved or were unchanged. No other regressions.
