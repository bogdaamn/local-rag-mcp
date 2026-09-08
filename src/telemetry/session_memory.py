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
