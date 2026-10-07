from src.embeddings import generate_embeddings
from src.retriever import embed_question, retrieve_relevant_chunks
from src.vector_store import create_faiss_index
from src.models import DocumentChunk

def build_test_retrieval_system():
    """Build a small deterministic retrieval system for testing."""
    chunks = [
    DocumentChunk(
        text="Python is a programming language widely used for data science.",
        source_file="test_document.pdf",
        page_number=1,
        chunk_id="page-001-chunk-001",
    ),
    DocumentChunk(
        text="Machine learning algorithms learn patterns from data.",
        source_file="test_document.pdf",
        page_number=2,
        chunk_id="page-002-chunk-002",
    ),
    DocumentChunk(
        text="FAISS is a library for efficient similarity search of embeddings.",
        source_file="test_document.pdf",
        page_number=3,
        chunk_id="page-003-chunk-003",
    ),
    DocumentChunk(
        text="Streamlit can be used to build interactive data applications.",
        source_file="test_document.pdf",
        page_number=4,
        chunk_id="page-004-chunk-004",
    ),
]

    embeddings = generate_embeddings(chunks)
    index = create_faiss_index(embeddings)

    return chunks, index



def test_retrieval_returns_python_chunk_for_python_question():
    chunks, index = build_test_retrieval_system()

    question_embedding = embed_question(
        "What programming language is widely used for data science?"
    )

    retrieved_chunks = retrieve_relevant_chunks(
        index,
        chunks,
        question_embedding,
        k=2,
    )

    assert "Python" in retrieved_chunks[0]["text"]


def test_retrieval_returns_machine_learning_chunk_for_ml_question():
    chunks, index = build_test_retrieval_system()

    question_embedding = embed_question(
        "How do machine learning algorithms learn?"
    )

    retrieved_chunks = retrieve_relevant_chunks(
        index,
        chunks,
        question_embedding,
        k=2,
    )

    assert "Machine learning" in retrieved_chunks[0]["text"]


def test_retrieval_returns_faiss_chunk_for_faiss_question():
    chunks, index = build_test_retrieval_system()

    question_embedding = embed_question(
        "Which library provides efficient similarity search for embeddings?"
    )

    retrieved_chunks = retrieve_relevant_chunks(
        index,
        chunks,
        question_embedding,
        k=2,
    )

    assert "FAISS" in retrieved_chunks[0]["text"]

def test_retrieval_returns_requested_top_k_results():
    chunks, index = build_test_retrieval_system()

    question_embedding = embed_question(
        "What is machine learning?"
    )

    retrieved_chunks = retrieve_relevant_chunks(
        index,
        chunks,
        question_embedding,
        k=3,
    )

    assert len(retrieved_chunks) == 3


def test_retrieval_results_are_deterministic():
    chunks, index = build_test_retrieval_system()

    question_embedding = embed_question(
        "What is FAISS used for?"
    )

    first_result = retrieve_relevant_chunks(
        index,
        chunks,
        question_embedding,
        k=2,
    )

    second_result = retrieve_relevant_chunks(
        index,
        chunks,
        question_embedding,
        k=2,
    )

    assert first_result == second_result