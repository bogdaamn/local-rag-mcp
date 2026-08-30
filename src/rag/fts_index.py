import sqlite3
from pathlib import Path


def build_fts_index(chunks, fts_db_path):
    """(Re)create the FTS5 virtual table from scratch and populate it from
    `chunks`. Overwrites any existing file at `fts_db_path`."""
    fts_db_path = Path(fts_db_path)
    if fts_db_path.exists():
        fts_db_path.unlink()

    conn = sqlite3.connect(str(fts_db_path))
    try:
        conn.execute(
            """
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                id UNINDEXED,
                text,
                tokenize = 'porter unicode61'
            )
            """
        )
        conn.executemany(
            "INSERT INTO chunks_fts (id, text) VALUES (?, ?)",
            [(c["id"], c["text"]) for c in chunks],
        )
        conn.commit()
    finally:
        conn.close()


def search_fts(fts_db_path, query, top_k):
    """Run one FTS5 MATCH query and return up to top_k chunk ids, best match
    first. Opens a new connection per call (SQLite connections aren't
    thread-safe; this is called concurrently from a ThreadPoolExecutor)."""
    tokens = query.split()
    if not tokens:
        return []
    match_expr = " OR ".join(f'"{token}"' for token in tokens)

    conn = sqlite3.connect(str(fts_db_path))
    try:
        cursor = conn.execute(
            "SELECT id FROM chunks_fts WHERE chunks_fts MATCH ? "
            "ORDER BY bm25(chunks_fts) LIMIT ?",
            (match_expr, top_k),
        )
        return [row[0] for row in cursor.fetchall()]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()
