from telemetry import llm_cache


def test_get_returns_none_on_miss(tmp_path):
    db_path = tmp_path / "telemetry.db"
    assert llm_cache.get("hello", "qwen3:0.6b", None, db_path=db_path) is None


def test_put_then_get_is_a_hit(tmp_path):
    db_path = tmp_path / "telemetry.db"
    llm_cache.put("hello world", "qwen3:0.6b", None, "the answer", 10, 5, db_path=db_path)

    result = llm_cache.get("hello world", "qwen3:0.6b", None, db_path=db_path)

    assert result == {"response": "the answer", "input_tokens": 10, "output_tokens": 5}


def test_normalization_collapses_whitespace_case_and_punctuation(tmp_path):
    db_path = tmp_path / "telemetry.db"
    llm_cache.put("Hello, World!", "qwen3:0.6b", None, "answer", 1, 1, db_path=db_path)

    assert llm_cache.get("  hello world  ", "qwen3:0.6b", None, db_path=db_path) is not None


def test_normalization_does_not_conflate_different_questions(tmp_path):
    db_path = tmp_path / "telemetry.db"
    llm_cache.put("what is the vacation policy", "qwen3:0.6b", None, "a1", 1, 1, db_path=db_path)

    assert llm_cache.get("what is the expense policy", "qwen3:0.6b", None, db_path=db_path) is None


def test_different_model_is_a_miss(tmp_path):
    db_path = tmp_path / "telemetry.db"
    llm_cache.put("q", "qwen3:0.6b", None, "a", 1, 1, db_path=db_path)

    assert llm_cache.get("q", "other-model", None, db_path=db_path) is None


def test_different_temperature_is_a_miss(tmp_path):
    db_path = tmp_path / "telemetry.db"
    llm_cache.put("q", "qwen3:0.6b", 0.1, "a", 1, 1, db_path=db_path)

    assert llm_cache.get("q", "qwen3:0.6b", 0.9, db_path=db_path) is None


def test_clear_removes_all_entries(tmp_path):
    db_path = tmp_path / "telemetry.db"
    llm_cache.put("q", "qwen3:0.6b", None, "a", 1, 1, db_path=db_path)

    llm_cache.clear(db_path=db_path)

    assert llm_cache.get("q", "qwen3:0.6b", None, db_path=db_path) is None


def test_put_overwrites_an_existing_key(tmp_path):
    db_path = tmp_path / "telemetry.db"
    llm_cache.put("q", "qwen3:0.6b", None, "first", 1, 1, db_path=db_path)
    llm_cache.put("q", "qwen3:0.6b", None, "second", 2, 2, db_path=db_path)

    assert llm_cache.get("q", "qwen3:0.6b", None, db_path=db_path) == {
        "response": "second", "input_tokens": 2, "output_tokens": 2,
    }
