import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"

embedder = SentenceTransformer(
    MODEL_NAME,
    local_files_only=True,
)


def generate_embeddings(chunks: list[str]):
    """Generate normalized embeddings for text chunks."""
    if not chunks:
        return []

    embeddings = embedder.encode(
        chunks,
        normalize_embeddings=True,
    )

    return embeddings.astype("float32")