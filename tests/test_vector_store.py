import numpy as np
import pytest

from src.vector_store import create_faiss_index


def test_create_faiss_index_stores_all_embeddings():
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype="float32",
    )

    index = create_faiss_index(embeddings)

    assert index.ntotal == 3


def test_create_faiss_index_uses_embedding_dimension():
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype="float32",
    )

    index = create_faiss_index(embeddings)

    assert index.d == 4


def test_create_faiss_index_rejects_empty_embeddings():
    embeddings = np.empty((0, 384), dtype="float32")

    with pytest.raises(
        ValueError,
        match="Cannot create a FAISS index from empty embeddings",
    ):
        create_faiss_index(embeddings)
