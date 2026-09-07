import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
# Reuses rag/expand.py's whitespace/case/punctuation collapse (per spec) so
# the cache and query-expansion dedup logic never drift apart.
from rag.expand import _normalize as normalize
from telemetry import storage


def _cache_key(prompt, model, temperature):
    raw = f"{normalize(prompt)}{model}{temperature}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get(prompt, model, temperature, db_path=None):
    """Look up a cached response. Returns a dict with response/input_tokens/
    output_tokens on hit, None on miss."""
    key = _cache_key(prompt, model, temperature)
    conn = storage.get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT response, input_tokens, output_tokens FROM llm_cache WHERE cache_key = ?",
            (key,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None
    return {"response": row[0], "input_tokens": row[1], "output_tokens": row[2]}


def put(prompt, model, temperature, response, input_tokens, output_tokens, db_path=None):
    key = _cache_key(prompt, model, temperature)
    conn = storage.get_connection(db_path)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO llm_cache "
            "(cache_key, model, response, input_tokens, output_tokens, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (key, model, response, input_tokens, output_tokens, storage.now_iso()),
        )
        conn.commit()
    finally:
        conn.close()


def clear(db_path=None):
    conn = storage.get_connection(db_path)
    try:
        conn.execute("DELETE FROM llm_cache")
        conn.commit()
    finally:
        conn.close()
