from mcp.decision_heuristic import decide_tool_usage


def test_list_documents_question_triggers_list_documents_with_no_args():
    tool, args = decide_tool_usage("List all the documents in the knowledge base.", [])
    assert (tool, args) == ("list_documents", {})


def test_which_documents_question_also_triggers_list_documents():
    tool, args = decide_tool_usage("Which documents exist about ranking?", [])
    assert tool == "list_documents"


def test_search_command_triggers_search_documents_with_extracted_topic():
    tool, args = decide_tool_usage("Search the knowledge base for anything about embeddings.", [])
    assert tool == "search_documents"
    assert args == {"query": "embeddings"}


def test_find_command_triggers_search_documents_with_extracted_topic():
    tool, args = decide_tool_usage("Find the document about sentence embeddings", [])
    assert tool == "search_documents"
    assert args == {"query": "sentence embeddings"}


def test_read_full_contents_of_named_file_triggers_read_document():
    tool, args = decide_tool_usage("Read the full contents of sqlite.txt.", [])
    assert tool == "read_document"
    assert args == {"file_path": "docs/sqlite.txt"}


def test_read_document_extracts_topic_when_about_is_present():
    tool, args = decide_tool_usage("Read the full contents of sqlite.txt about WAL mode.", [])
    assert tool == "read_document"
    assert args == {"file_path": "docs/sqlite.txt", "query": "WAL mode"}


def test_read_without_a_recognizable_filename_uses_no_tool():
    tool, args = decide_tool_usage("Can you read me the contents of the policy?", [])
    assert (tool, args) == (None, None)


def test_pure_rag_question_with_no_document_keyword_uses_no_tool():
    tool, args = decide_tool_usage("What is BM25 and how does it use avgdl in ranking?", [])
    assert (tool, args) == (None, None)


def test_search_word_in_a_conceptual_question_does_not_trigger_a_tool():
    tool, args = decide_tool_usage(
        "What is FTS5 in SQLite and how does it relate to full-text search?", [],
    )
    assert (tool, args) == (None, None)


def test_cross_document_synthesis_question_uses_no_tool():
    tool, args = decide_tool_usage(
        "How do sentence embeddings differ from BM25 ranking for search relevance?", [],
    )
    assert (tool, args) == (None, None)


def test_mixed_intent_question_resolves_to_list_documents():
    tool, args = decide_tool_usage(
        "What was Robertson's contribution to BM25, and what other documents "
        "exist about ranking or search?", [],
    )
    assert tool == "list_documents"


def test_contexts_argument_is_accepted_but_does_not_affect_the_decision():
    tool_a, args_a = decide_tool_usage("avgdl", [])
    tool_b, args_b = decide_tool_usage("avgdl", [{"source": "x", "text": "y"}])
    assert (tool_a, args_a) == (tool_b, args_b) == (None, None)
