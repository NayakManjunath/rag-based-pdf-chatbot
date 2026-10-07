from src.rag_pipeline import build_context


def test_build_context_combines_retrieved_chunks():
    chunks = [
        {
            "text": "First document section.",
            "source_file": "test.pdf",
            "page_number": 1,
            "chunk_id": "page-001-chunk-001",
            "score": 0.95,
        },
        {
            "text": "Second document section.",
            "source_file": "test.pdf",
            "page_number": 2,
            "chunk_id": "page-002-chunk-001",
            "score": 0.90,
        },
        {
            "text": "Third document section.",
            "source_file": "test.pdf",
            "page_number": 3,
            "chunk_id": "page-003-chunk-001",
            "score": 0.85,
        },
    ]

    context = build_context(chunks)

    expected = (
        "First document section.\n\n"
        "Second document section.\n\n"
        "Third document section."
    )

    assert context == expected


def test_build_context_returns_empty_string_for_empty_chunks():
    assert build_context([]) == ""


def test_build_context_preserves_chunk_content():
    chunks = [
        {
            "text": "Working hours are 9 AM to 5 PM.",
            "source_file": "employee_handbook.pdf",
            "page_number": 4,
            "chunk_id": "page-004-chunk-001",
            "score": 0.92,
        },
        {
            "text": "Employees must carry their ID card.",
            "source_file": "employee_handbook.pdf",
            "page_number": 5,
            "chunk_id": "page-005-chunk-001",
            "score": 0.88,
        },
    ]

    context = build_context(chunks)

    assert "Working hours are 9 AM to 5 PM." in context
    assert "Employees must carry their ID card." in context