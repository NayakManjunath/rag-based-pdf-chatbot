import numpy as np

from src.embeddings import generate_embeddings
from src.models import DocumentChunk


def test_generate_embeddings_returns_one_embedding_per_chunk():
    chunks = [
        DocumentChunk(
            text="Employees must carry their ID card.",
            source_file="test.pdf",
            page_number=1,
            chunk_id="page-001-chunk-001",
        ),
        DocumentChunk(
            text="Leave policy applies to all employees.",
            source_file="test.pdf",
            page_number=2,
            chunk_id="page-002-chunk-001",
        ),
        DocumentChunk(
            text="Working hours are from 9 AM to 5 PM.",
            source_file="test.pdf",
            page_number=3,
            chunk_id="page-003-chunk-001",
        ),
    ]

    embeddings = generate_embeddings(chunks)

    assert embeddings.shape == (3, 384)


def test_generate_embeddings_returns_float32():
    chunks = [
        DocumentChunk(
            text="This is a test document.",
            source_file="test.pdf",
            page_number=1,
            chunk_id="page-001-chunk-001",
        ),
        DocumentChunk(
            text="This is another test document.",
            source_file="test.pdf",
            page_number=2,
            chunk_id="page-002-chunk-001",
        ),
    ]

    embeddings = generate_embeddings(chunks)

    assert embeddings.dtype == np.float32


def test_generate_embeddings_are_normalized():
    chunks = [
        DocumentChunk(
            text="Employees must carry their ID card.",
            source_file="test.pdf",
            page_number=1,
            chunk_id="page-001-chunk-001",
        ),
        DocumentChunk(
            text="Leave policy applies to all employees.",
            source_file="test.pdf",
            page_number=2,
            chunk_id="page-002-chunk-001",
        ),
    ]

    embeddings = generate_embeddings(chunks)

    norms = np.linalg.norm(embeddings, axis=1)

    assert np.allclose(norms, 1.0, atol=1e-5)


def test_generate_embeddings_returns_empty_list_for_empty_input():
    assert generate_embeddings([]) == []