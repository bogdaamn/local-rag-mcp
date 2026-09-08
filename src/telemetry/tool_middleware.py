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
