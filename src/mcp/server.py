import json
from fastmcp import FastMCP
from pathlib import Path
import sys

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DOCUMENTS_DIR, READ_DOCUMENT_MAX_CHARS

mcp = FastMCP("doc-tools", version="1.0.0")


def _matching_excerpt(text, query, context_lines=2):
    """Grep-like: lines containing `query` (case-insensitive substring),
    each surrounded by `context_lines` lines of context; overlapping
    ranges are merged, non-adjacent ranges are joined with a separator.
    Returns None if `query` matches no line."""
    lines = text.splitlines()
    query_lower = query.lower()
    matches = [i for i, line in enumerate(lines) if query_lower in line.lower()]

    if not matches:
        return None

    ranges = []
    for i in matches:
        start = max(0, i - context_lines)
        end = min(len(lines), i + context_lines + 1)
        if ranges and start <= ranges[-1][1]:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
        else:
            ranges.append((start, end))

    parts = ["\n".join(lines[start:end]) for start, end in ranges]
    return "\n...\n".join(parts)


@mcp.tool
def read_document(file_path: str, query: str = None, max_chars: int = None, offset: int = 0) -> str:
    """Reads a document from the knowledge base. Without `query`, returns a
    budgeted window of the whole file (text[offset:offset+max_chars],
    default budget READ_DOCUMENT_MAX_CHARS), reporting truncation and a
    next_offset for continuation. With `query`, returns only the lines
    matching it (case-insensitive substring) plus surrounding context,
    instead of the whole file. On success, returns a JSON-encoded string
    {"content", "truncated", "total_chars", "next_offset"}. On error,
    returns a plain (non-JSON) string, same as the other 2 tools."""
    try:
        path = Path(file_path)
        # Security: ensure path is within documents directory
        if not str(path.resolve()).startswith(str(Path(DOCUMENTS_DIR).resolve())):
            return f"Error: Access denied. File must be in {DOCUMENTS_DIR}"

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        effective_max = max_chars or READ_DOCUMENT_MAX_CHARS

        if query:
            excerpt = _matching_excerpt(text, query)
            if excerpt is None:
                return json.dumps({
                    "content": f"No lines matching '{query}' found in {file_path}.",
                    "truncated": False, "total_chars": len(text), "next_offset": None,
                })
            truncated = len(excerpt) > effective_max
            return json.dumps({
                "content": excerpt[:effective_max],
                "truncated": truncated, "total_chars": len(excerpt), "next_offset": None,
            })

        window = text[offset:offset + effective_max]
        truncated = offset + len(window) < len(text)
        return json.dumps({
            "content": window,
            "truncated": truncated,
            "total_chars": len(text),
            "next_offset": offset + effective_max if truncated else None,
        })
    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except Exception as e:
        return f"Error reading file: {str(e)}"


@mcp.tool
def list_documents() -> str:
    """Lists all available documents in the knowledge base."""
    try:
        base_dir = Path(DOCUMENTS_DIR)
        if not base_dir.exists():
            return f"Error: Documents directory {DOCUMENTS_DIR} does not exist"

        documents = []
        for path in base_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".txt", ".md", ".pdf", ".docx"}:
                documents.append(str(path.relative_to(base_dir)))

        if not documents:
            return "No documents found in the knowledge base."

        return "\n".join(f"- {doc}" for doc in sorted(documents))
    except Exception as e:
        return f"Error listing documents: {str(e)}"


@mcp.tool
def search_documents(query: str) -> str:
    """Searches for documents by name (case-insensitive)."""
    try:
        base_dir = Path(DOCUMENTS_DIR)
        if not base_dir.exists():
            return f"Error: Documents directory {DOCUMENTS_DIR} does not exist"

        query_lower = query.lower()
        matches = []

        for path in base_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".txt", ".md", ".pdf", ".docx"}:
                if query_lower in path.name.lower():
                    matches.append(str(path.relative_to(base_dir)))

        if not matches:
            return f"No documents found matching '{query}'"

        return "\n".join(f"- {doc}" for doc in sorted(matches))
    except Exception as e:
        return f"Error searching documents: {str(e)}"


if __name__ == "__main__":
    # Run MCP server (stdio)
    mcp.run()
