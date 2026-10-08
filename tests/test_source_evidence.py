from src.source_evidence import format_source_evidence


def test_format_source_evidence_returns_source_metadata():
    retrieved_chunks = [
        {
            "text": "The eDAR system supports accident data entry.",
            "source_file": "test_document.pdf",
            "page_number": 5,
            "chunk_id": "page-005-chunk-001",
            "score": 0.92,
        }
    ]

    sources = format_source_evidence(retrieved_chunks)

    assert sources == [
        {
            "source_file": "test_document.pdf",
            "page_number": 5,
            "chunk_id": "page-005-chunk-001",
        }
    ]


def test_format_source_evidence_supports_multiple_sources():
    retrieved_chunks = [
        {
            "text": "First relevant section.",
            "source_file": "test_document.pdf",
            "page_number": 2,
            "chunk_id": "page-002-chunk-001",
            "score": 0.95,
        },
        {
            "text": "Second relevant section.",
            "source_file": "test_document.pdf",
            "page_number": 7,
            "chunk_id": "page-007-chunk-002",
            "score": 0.89,
        },
    ]

    sources = format_source_evidence(retrieved_chunks)

    assert len(sources) == 2
    assert sources[0]["source_file"] == "test_document.pdf"
    assert sources[0]["page_number"] == 2
    assert sources[0]["chunk_id"] == "page-002-chunk-001"

    assert sources[1]["source_file"] == "test_document.pdf"
    assert sources[1]["page_number"] == 7
    assert sources[1]["chunk_id"] == "page-007-chunk-002"


def test_format_source_evidence_returns_empty_list_for_no_results():
    assert format_source_evidence([]) == []


def test_format_source_evidence_does_not_expose_retrieval_score():
    retrieved_chunks = [
        {
            "text": "Relevant document section.",
            "source_file": "employee_handbook.pdf",
            "page_number": 4,
            "chunk_id": "page-004-chunk-001",
            "score": 0.97,
        }
    ]

    sources = format_source_evidence(retrieved_chunks)

    assert "score" not in sources[0]


def test_format_source_evidence_preserves_chunk_identity():
    retrieved_chunks = [
        {
            "text": "Important section.",
            "source_file": "employee_handbook.pdf",
            "page_number": 12,
            "chunk_id": "page-012-chunk-003",
            "score": 0.91,
        }
    ]

    sources = format_source_evidence(retrieved_chunks)

    assert sources[0]["source_file"] == "employee_handbook.pdf"
    assert sources[0]["page_number"] == 12
    assert sources[0]["chunk_id"] == "page-012-chunk-003"