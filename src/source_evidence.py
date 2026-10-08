def format_source_evidence(retrieved_chunks: list[dict]) -> list[dict]:
    """
    Extract user-facing source evidence from retrieved chunks.

    Each source contains the PDF filename, page number,
    and chunk ID needed to trace the retrieved evidence.
    """
    if not retrieved_chunks:
        return []

    sources = []

    for chunk in retrieved_chunks:
        sources.append(
            {
                "source_file": chunk["source_file"],
                "page_number": chunk["page_number"],
                "chunk_id": chunk["chunk_id"],
            }
        )

    return sources