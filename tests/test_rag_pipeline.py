from src.rag_pipeline import build_context


def test_build_context_combines_retrieved_chunks():
    chunks = [
        "First document section.",
        "Second document section.",
        "Third document section.",
    ]

    context = build_context(chunks)

    assert context == (
        "First document section.\n\n"
        "Second document section.\n\n"
        "Third document section."
    )


def test_build_context_returns_empty_string_for_empty_chunks():
    assert build_context([]) == ""


def test_build_context_preserves_chunk_content():
    chunks = [
        "Working hours are 9 AM to 5 PM.",
        "Employees must carry their ID card.",
    ]

    context = build_context(chunks)

    assert "Working hours are 9 AM to 5 PM." in context
    assert "Employees must carry their ID card." in context
