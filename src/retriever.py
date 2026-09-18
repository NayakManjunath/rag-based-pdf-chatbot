from src.embeddings import generate_embeddings


def embed_question(question: str):
    """Generate an embedding for a user question."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    return generate_embeddings([question.strip()])[0]


def retrieve_relevant_chunks(index, chunks: list[str], question_embedding, k: int = 3):
    """Retrieve the most relevant chunks for a question."""
    if not chunks:
        return []

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    k = min(k, len(chunks))

    query_vector = question_embedding.reshape(1, -1)

    scores, indices = index.search(query_vector, k)

    return [
        chunks[index_position]
        for index_position in indices[0]
        if index_position != -1
    ]