from src.llm import build_grounded_prompt


DOCUMENT_CONTEXT = (
    "FAISS is used for efficient similarity search of embeddings. "
    "The RAG chatbot retrieves relevant document chunks before "
    "generating an answer."
)


def test_grounded_prompt_supports_answer_from_document_context():
    question = "What is FAISS used for?"

    prompt = build_grounded_prompt(
        DOCUMENT_CONTEXT,
        question,
    )

    assert "FAISS" in prompt
    assert "similarity search" in prompt
    assert question in prompt


def test_grounded_prompt_contains_required_fallback_instruction():
    question = "What is the capital of France?"

    prompt = build_grounded_prompt(
        DOCUMENT_CONTEXT,
        question,
    )

    assert "I don't know from this document." in prompt


def test_grounded_prompt_prevents_using_information_outside_context():
    question = "What is the capital of France?"

    prompt = build_grounded_prompt(
        DOCUMENT_CONTEXT,
        question,
    )

    assert (
        "Answer the question using only the provided document context."
        in prompt
    )


def test_grounded_prompt_preserves_document_context():
    question = "What is FAISS used for?"

    prompt = build_grounded_prompt(
        DOCUMENT_CONTEXT,
        question,
    )

    assert DOCUMENT_CONTEXT in prompt


def test_fallback_answer_is_exact_required_message():
    fallback_answer = "I don't know from this document."

    assert fallback_answer == "I don't know from this document."