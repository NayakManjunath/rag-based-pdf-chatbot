import numpy as np

from src.embeddings import generate_embeddings
from src.models import DocumentChunk


def embed_question(question: str):
    """Generate an embedding for a user question."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    return generate_embeddings(
        [
            DocumentChunk(
                text=question.strip(),
                source_file="",
                page_number=0,
                chunk_id="query",
            )
        ]
    )[0]


def retrieve_relevant_chunks(
    index,
    chunks: list[DocumentChunk],
    question_embedding,
    k: int = 3,
):
    """Retrieve relevant document chunks while preserving metadata."""
    if not chunks:
        return []

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    k = min(k, len(chunks))

    query_vector = np.asarray(
        question_embedding,
        dtype="float32",
    ).reshape(1, -1)

    scores, indices = index.search(query_vector, k)

    results = []

    for score, index_position in zip(
        scores[0],
        indices[0],
    ):
        if index_position == -1:
            continue

        chunk = chunks[index_position]

        results.append(
            {
                "text": chunk.text,
                "source_file": chunk.source_file,
                "page_number": chunk.page_number,
                "chunk_id": chunk.chunk_id,
                "score": float(score),
            }
        )

    return results

def retrieve_with_conversation_context(
    index,
    chunks: list[DocumentChunk],
    chat_history: list[dict],
    question: str,
    k: int = 3,
) -> list[dict]:
    """
    Resolve a conversational question and retrieve relevant document chunks.

    The resolved question is used only for retrieval.
    The original user question remains unchanged in chat history.
    """
    from src.query_resolver import resolve_question

    resolved_question = resolve_question(
        chat_history,
        question,
    )

    question_embedding = embed_question(resolved_question)

    return retrieve_relevant_chunks(
        index,
        chunks,
        question_embedding,
        k=k,
    )