# tests/test_config.py
import pytest

import config


@pytest.fixture(autouse=True)
def isolated_telemetry_db():
    """Override the global autouse fixture (tests/conftest.py) with a no-op
    for this module: these tests assert the raw, unmodified config constants,
    so telemetry-path redirection must not apply here."""
    yield


def test_hybrid_search_config_constants():
    assert config.FTS_DB_PATH == "fts.db"
    assert config.RRF_K == 60
    assert config.NUM_QUERY_EXPANSIONS == 3
    assert config.CANDIDATE_K == config.TOP_K * 3


def test_telemetry_config_constants():
    assert config.TELEMETRY_DB_PATH == "telemetry.db"
    assert config.AGENT_ID == "company-kb-assistant"


def test_read_document_max_chars_default():
    assert config.READ_DOCUMENT_MAX_CHARS == 4000
