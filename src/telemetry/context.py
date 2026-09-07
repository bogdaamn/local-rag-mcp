import contextvars
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import AGENT_ID

current_agent_id = contextvars.ContextVar("current_agent_id", default=AGENT_ID)
current_task_id = contextvars.ContextVar("current_task_id", default=None)
current_turn_number = contextvars.ContextVar("current_turn_number", default=0)


def start_task(task_id=None):
    """Begin a new task: generate (or accept) a task_id, reset the turn
    counter to 0, and set both as the ambient context. Returns the task_id."""
    task_id = task_id or str(uuid.uuid4())
    current_task_id.set(task_id)
    current_turn_number.set(0)
    return task_id


def next_turn():
    """Advance to the next turn within the current task. Returns the new
    turn number."""
    turn = current_turn_number.get() + 1
    current_turn_number.set(turn)
    return turn
