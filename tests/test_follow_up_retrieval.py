import numpy as np

from src.models import DocumentChunk
from src.retriever import retrieve_with_conversation_context


def test_resolved_follow_up_question_reaches_embedding_and_retrieval(
    monkeypatch,
):
    history = [
        {
            "role": "user",
            "content": "What is eDAR?",
        },
        {
            "role": "assistant",
            "content": "eDAR is an electronic accident reporting system.",
            "sources": [],
        },
    ]

    chunks = [
        DocumentChunk(
            text="eDAR is used by hospital staff.",
            source_file="employee_handbook.pdf",
            page_number=2,
            chunk_id="page-002-chunk-001",
        ),
        DocumentChunk(
            text="The hospital module manages hospital-related reporting.",
            source_file="employee_handbook.pdf",
            page_number=3,
            chunk_id="page-003-chunk-002",
        ),
    ]

    captured = {}

    def fake_resolve_question(chat_history, question):
        captured["history"] = chat_history
        captured["question"] = question
        return "Who uses eDAR?"

    def fake_embed_question(question):
        captured["embedded_question"] = question
        return np.array([1.0, 0.0], dtype="float32")

    def fake_retrieve_relevant_chunks(
        index,
        chunks,
        question_embedding,
        k=3,
    ):
        captured["question_embedding"] = question_embedding
        captured["k"] = k

        return [
            {
                "text": "eDAR is used by hospital staff.",
                "source_file": "employee_handbook.pdf",
                "page_number": 2,
                "chunk_id": "page-002-chunk-001",
                "score": 0.95,
            }
        ]

    monkeypatch.setattr(
        "src.query_resolver.resolve_question",
        fake_resolve_question,
    )

    monkeypatch.setattr(
        "src.retriever.embed_question",
        fake_embed_question,
    )

    monkeypatch.setattr(
        "src.retriever.retrieve_relevant_chunks",
        fake_retrieve_relevant_chunks,
    )

    results = retrieve_with_conversation_context(
        index=object(),
        chunks=chunks,
        chat_history=history,
        question="Who uses it?",
        k=3,
    )

    assert captured["history"] == history
    assert captured["question"] == "Who uses it?"

    assert captured["embedded_question"] == "Who uses eDAR?"

    assert captured["question_embedding"].tolist() == [1.0, 0.0]

    assert captured["k"] == 3

    assert results[0]["text"] == "eDAR is used by hospital staff."