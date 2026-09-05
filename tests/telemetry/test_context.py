import uuid

from telemetry import context


def test_current_agent_id_defaults_to_config_agent_id():
    assert context.current_agent_id.get() == "company-kb-assistant"


def test_start_task_generates_a_uuid4_and_resets_turn_number():
    context.current_turn_number.set(7)

    task_id = context.start_task()

    assert uuid.UUID(task_id).version == 4
    assert context.current_task_id.get() == task_id
    assert context.current_turn_number.get() == 0


def test_start_task_accepts_an_explicit_task_id():
    task_id = context.start_task("explicit-id-123")

    assert task_id == "explicit-id-123"
    assert context.current_task_id.get() == "explicit-id-123"


def test_next_turn_increments_from_the_current_value():
    context.start_task("t1")

    assert context.next_turn() == 1
    assert context.next_turn() == 2
    assert context.current_turn_number.get() == 2


def test_nested_calls_see_the_ambient_context():
    context.start_task("t2")
    context.next_turn()

    def inner():
        return context.current_task_id.get(), context.current_turn_number.get()

    assert inner() == ("t2", 1)
