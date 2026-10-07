from sentence_transformers import SentenceTransformer

from src.models import DocumentChunk


MODEL_NAME = "all-MiniLM-L6-v2"

embedder = SentenceTransformer(
    MODEL_NAME,
)


def generate_embeddings(chunks: list[DocumentChunk]):
    """Generate normalized embeddings using chunk text only."""
    if not chunks:
        return []

    texts = [chunk.text for chunk in chunks]

    embeddings = embedder.encode(
        texts,
        normalize_embeddings=True,
    )

    return embeddings.astype("float32")