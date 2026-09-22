import numpy as np

from src.embeddings import generate_embeddings


def test_generate_embeddings_returns_one_embedding_per_chunk():
    chunks = [
        "Employees must carry their ID card.",
        "Leave policy applies to all employees.",
        "Working hours are from 9 AM to 5 PM.",
    ]

    embeddings = generate_embeddings(chunks)

    assert embeddings.shape == (3, 384)


def test_generate_embeddings_returns_float32():
    chunks = [
        "This is a test document.",
        "This is another test document.",
    ]

    embeddings = generate_embeddings(chunks)

    assert embeddings.dtype == np.float32


def test_generate_embeddings_are_normalized():
    chunks = [
        "Employees must carry their ID card.",
        "Leave policy applies to all employees.",
    ]

    embeddings = generate_embeddings(chunks)

    norms = np.linalg.norm(embeddings, axis=1)

    assert np.allclose(norms, 1.0, atol=1e-5)


def test_generate_embeddings_returns_empty_list_for_empty_input():
    assert generate_embeddings([]) == []
