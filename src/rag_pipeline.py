def build_context(retrieved_chunks: list[dict]) -> str:
    """Combine retrieved chunk text into a single RAG context."""
    if not retrieved_chunks:
        return ""

    return "\n\n".join(
        chunk["text"]
        for chunk in retrieved_chunks
    )