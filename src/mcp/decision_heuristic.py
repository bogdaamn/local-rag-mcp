import re

# Matches config.DOCUMENTS_DIR = "./docs", without the leading "./" — the
# assembled file_path is what gets passed to read_document's own path
# resolution, so it must match that convention.
DOCUMENTS_PATH_PREFIX = "docs"

FILENAME_RE = re.compile(r"\b([\w\-]+\.(?:txt|md|pdf|docx))\b", re.IGNORECASE)


def _extract_topic(query):
    """Best-effort topic extraction for search_documents'/read_document's
    `query` arg: text after the last ' about ' or ' for ', else the
    query's last word. Approximate on purpose — deterministic heuristic,
    not an LLM."""
    lower = query.lower()
    for marker in (" about ", " for "):
        if marker in lower:
            idx = lower.rindex(marker) + len(marker)
            topic = query[idx:].strip().rstrip("?.!")
            if topic:
                return topic
    words = query.strip().rstrip("?.!").split()
    return words[-1] if words else query


def decide_tool_usage(query, contexts):
    """Deterministic, rule-based replacement for the old LLM-based
    _llm_decide_mcp_usage/mcp_decision call. Same return contract as the
    method it replaces: (tool_name, args) if a tool should be used, else
    (None, None). `contexts` is accepted but unused — kept only so this is
    a drop-in-compatible signature for assistant.py::query()'s one call
    site."""
    lower = query.strip().lower()

    if lower.startswith(("list", "which", "what")) and "document" in lower:
        return "list_documents", {}

    if lower.startswith(("search", "find")):
        return "search_documents", {"query": _extract_topic(query)}

    filename_match = FILENAME_RE.search(query)
    if "read" in lower and ("full contents" in lower or "contents of" in lower or filename_match):
        if not filename_match:
            return None, None
        args = {"file_path": f"{DOCUMENTS_PATH_PREFIX}/{filename_match.group(1)}"}
        if " about " in lower:
            args["query"] = _extract_topic(query)
        return "read_document", args

    return None, None
