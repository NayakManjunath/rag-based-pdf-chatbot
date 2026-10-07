import numpy as np
import pytest
from src.models import DocumentChunk
from src.retriever import embed_question, retrieve_relevant_chunks
from src.vector_store import create_faiss_index
from src.models import DocumentChunk

def test_embed_question_returns_normalized_embedding():
    embedding = embed_question("What are the working hours?")

    assert embedding.shape == (384,)
    assert embedding.dtype == np.float32
    assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5)


def test_embed_question_strips_whitespace():
    embedding = embed_question("  What are the working hours?  ")

    assert embedding.shape == (384,)


def test_embed_question_rejects_empty_question():
    with pytest.raises(ValueError, match="Question cannot be empty"):
        embed_question("")


def test_retrieve_relevant_chunks_returns_top_k_chunks():
    chunks = [
        DocumentChunk(
            text="Employees must carry their ID card.",
            source_file="employee_handbook.pdf",
            page_number=1,
            chunk_id="page-001-chunk-001",
        ),
        DocumentChunk(
            text="Leave policy applies to all employees.",
            source_file="employee_handbook.pdf",
            page_number=2,
            chunk_id="page-002-chunk-001",
        ),
        DocumentChunk(
            text="Working hours are from 9 AM to 5 PM.",
            source_file="employee_handbook.pdf",
            page_number=3,
            chunk_id="page-003-chunk-001",
        ),
    ]

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.2],
        ],
        dtype="float32",
    )

    index = create_faiss_index(embeddings)

    question_embedding = np.array(
        [0.9, 0.1],
        dtype="float32",
    )

    retrieved = retrieve_relevant_chunks(
        index,
        chunks,
        question_embedding,
        k=2,
    )

    assert len(retrieved) == 2

    retrieved_text = [
        result["text"]
        for result in retrieved
    ]

    assert all(
        text in [chunk.text for chunk in chunks]
        for text in retrieved_text
    )

def test_retrieve_relevant_chunks_limits_k_to_available_chunks():
    chunks = [
        DocumentChunk(
            text="First chunk.",
            source_file="test.pdf",
            page_number=1,
            chunk_id="page-001-chunk-001",
        ),
        DocumentChunk(
            text="Second chunk.",
            source_file="test.pdf",
            page_number=2,
            chunk_id="page-002-chunk-001",
        ),
    ]

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype="float32",
    )

    index = create_faiss_index(embeddings)

    question_embedding = np.array(
        [1.0, 0.0],
        dtype="float32",
    )

    retrieved = retrieve_relevant_chunks(
        index,
        chunks,
        question_embedding,
        k=5,
    )

    assert len(retrieved) == 2


def test_retrieve_relevant_chunks_returns_empty_for_empty_chunks():
    embeddings = np.array(
        [
            [1.0, 0.0],
        ],
        dtype="float32",
    )

    index = create_faiss_index(embeddings)

    question_embedding = np.array(
        [1.0, 0.0],
        dtype="float32",
    )

    assert retrieve_relevant_chunks(
        index,
        [],
        question_embedding,
        k=3,
    ) == []


def test_retrieve_relevant_chunks_rejects_invalid_k():
    chunks = ["First chunk."]

    embeddings = np.array(
        [
            [1.0, 0.0],
        ],
        dtype="float32",
    )

    index = create_faiss_index(embeddings)

    question_embedding = np.array(
        [1.0, 0.0],
        dtype="float32",
    )

    with pytest.raises(ValueError, match="k must be greater than zero"):
        retrieve_relevant_chunks(
            index,
            chunks,
            question_embedding,
            k=0,
        )

def test_retrieve_relevant_chunks_preserves_metadata():
    chunks = [
        DocumentChunk(
            text="Employees must carry their ID card.",
            source_file="employee_handbook.pdf",
            page_number=1,
            chunk_id="page-001-chunk-001",
        ),
        DocumentChunk(
            text="Leave policy applies to all employees.",
            source_file="employee_handbook.pdf",
            page_number=2,
            chunk_id="page-002-chunk-001",
        ),
        DocumentChunk(
            text="Working hours are from 9 AM to 5 PM.",
            source_file="employee_handbook.pdf",
            page_number=3,
            chunk_id="page-003-chunk-001",
        ),
    ]

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.2],
        ],
        dtype="float32",
    )

    index = create_faiss_index(embeddings)

    question_embedding = np.array(
        [0.9, 0.1],
        dtype="float32",
    )

    retrieved = retrieve_relevant_chunks(
        index,
        chunks,
        question_embedding,
        k=2,
    )

    assert len(retrieved) == 2

    assert retrieved[0]["source_file"] == "employee_handbook.pdf"
    assert retrieved[0]["page_number"] == 1
    assert retrieved[0]["chunk_id"] == "page-001-chunk-001"
    assert retrieved[0]["text"] == (
        "Employees must carry their ID card."
    )