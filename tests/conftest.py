import pytest


@pytest.fixture(autouse=True)
def isolated_telemetry_db(tmp_path, monkeypatch):
    """Redirect every test's telemetry writes to a per-test tmp file so test
    runs never touch (or get polluted by) the real project telemetry.db.
    storage.py re-imports config.TELEMETRY_DB_PATH locally on every call, so
    this monkeypatch takes effect even for code that doesn't accept a
    db_path override."""
    monkeypatch.setattr("config.TELEMETRY_DB_PATH", str(tmp_path / "telemetry.db"))
