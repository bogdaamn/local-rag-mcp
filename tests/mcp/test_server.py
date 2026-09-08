import json

import pytest

from mcp.server import read_document, list_documents, search_documents


@pytest.fixture
def docs_dir(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "sample.txt").write_text(
        "\n".join(f"line {i}" for i in range(1, 11)).replace("line 5", "line 5 mentions avgdl")
    )
    monkeypatch.setattr("mcp.server.DOCUMENTS_DIR", str(docs))
    return docs


def test_read_document_whole_file_under_budget_is_not_truncated(docs_dir):
    result = json.loads(read_document(str(docs_dir / "sample.txt")))
    assert result["truncated"] is False
    assert result["next_offset"] is None
    assert "line 1" in result["content"]
    assert "mentions avgdl" in result["content"]


def test_read_document_whole_file_over_budget_is_truncated_with_next_offset(docs_dir):
    result = json.loads(read_document(str(docs_dir / "sample.txt"), max_chars=10))
    assert result["truncated"] is True
    assert result["next_offset"] == 10
    assert len(result["content"]) == 10


def test_read_document_offset_continues_from_next_offset(docs_dir):
    first = json.loads(read_document(str(docs_dir / "sample.txt"), max_chars=10))
    second = json.loads(
        read_document(str(docs_dir / "sample.txt"), max_chars=10, offset=first["next_offset"])
    )
    full_text = (docs_dir / "sample.txt").read_text()
    assert first["content"] + second["content"] == full_text[:20]


def test_read_document_with_query_returns_only_matching_lines_with_context(docs_dir):
    result = json.loads(read_document(str(docs_dir / "sample.txt"), query="avgdl"))
    assert "mentions avgdl" in result["content"]
    assert "line 9" not in result["content"]
    assert result["truncated"] is False


def test_read_document_with_query_no_match_reports_zero_hits(docs_dir):
    result = json.loads(read_document(str(docs_dir / "sample.txt"), query="nonexistent-term"))
    assert "No lines matching" in result["content"]
    assert result["truncated"] is False


def test_read_document_with_query_merges_nearby_matches_into_one_range(tmp_path, monkeypatch):
    docs = tmp_path / "docs2"
    docs.mkdir()
    (docs / "merge.txt").write_text("a\nb avgdl\nc\nd avgdl\ne\nf\n")
    monkeypatch.setattr("mcp.server.DOCUMENTS_DIR", str(docs))

    result = json.loads(read_document(str(docs / "merge.txt"), query="avgdl"))

    assert "\n...\n" not in result["content"]
    assert result["content"].count("avgdl") == 2


def test_read_document_file_not_found_returns_plain_error_string(docs_dir):
    missing = docs_dir / "missing.txt"
    result = read_document(str(missing))
    assert result == f"Error: File not found: {missing}"


def test_read_document_access_denied_outside_documents_dir(docs_dir, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    result = read_document(str(outside))
    assert result.startswith("Error: Access denied")


def test_list_documents_and_search_documents_are_unaffected(docs_dir):
    assert "sample.txt" in list_documents()
    assert "sample.txt" in search_documents("sample")
