
from src.rag_pipeline import build_context
from src.source_evidence import format_source_evidence


def test_answer_context_and_displayed_sources_use_same_retrieved_chunks():
    retrieved_chunks = [
        {
            "text": "eDAR is an electronic accident reporting system.",
            "source_file": "employee_handbook.pdf",
            "page_number": 2,
            "chunk_id": "page-002-chunk-001",
            "score": 0.95,
        },
        {
            "text": "Hospital staff use eDAR to report accident information.",
            "source_file": "employee_handbook.pdf",
            "page_number": 3,
            "chunk_id": "page-003-chunk-002",
            "score": 0.89,
        },
    ]

    context = build_context(retrieved_chunks)
    sources = format_source_evidence(retrieved_chunks)

    # The answer context must contain the text from both retrieved chunks.
    assert "eDAR is an electronic accident reporting system." in context
    assert "Hospital staff use eDAR to report accident information." in context

    # Displayed source evidence must match the retrieved chunks.
    assert sources == [
        {
            "source_file": "employee_handbook.pdf",
            "page_number": 2,
            "chunk_id": "page-002-chunk-001",
        },
        {
            "source_file": "employee_handbook.pdf",
            "page_number": 3,
            "chunk_id": "page-003-chunk-002",
        },
    ]

    # Retrieval scores must not be exposed as citation metadata.
    assert all("score" not in source for source in sources)