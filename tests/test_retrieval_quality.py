from src.embeddings import generate_embeddings
from src.retriever import embed_question, retrieve_relevant_chunks
from src.vector_store import create_faiss_index


def build_test_retrieval_system():
    """Build a small deterministic retrieval system for testing."""
    chunks = [
        "Python is a programming language widely used for data science.",
        "Machine learning algorithms learn patterns from data.",
        "FAISS is a library for efficient similarity search of embeddings.",
        "Streamlit can be used to build interactive data applications.",
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

    assert chunks[0] in retrieved_chunks


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

    assert chunks[1] in retrieved_chunks


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

    assert chunks[2] in retrieved_chunks


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