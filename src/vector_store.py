import faiss


def create_faiss_index(embeddings):
    """Create a FAISS inner-product index from embeddings."""
    if len(embeddings) == 0:
        raise ValueError("Cannot create a FAISS index from empty embeddings.")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index