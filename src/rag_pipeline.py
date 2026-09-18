def build_context(retrieved_chunks: list[str]) -> str:
    """Combine retrieved PDF chunks into a single RAG context."""
    if not retrieved_chunks:
        return ""

    return "\n\n".join(retrieved_chunks)