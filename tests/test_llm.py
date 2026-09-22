import pytest

import src.llm as llm


def test_build_grounded_prompt_contains_context_and_question():
    context = "Working hours are Monday to Friday, 9 AM to 6 PM."
    question = "What are the working hours?"

    prompt = llm.build_grounded_prompt(
        context,
        question,
    )

    assert context in prompt
    assert question in prompt


def test_build_grounded_prompt_contains_grounding_instruction():
    context = "Employees must carry their ID card."
    question = "What must employees carry?"

    prompt = llm.build_grounded_prompt(
        context,
        question,
    )

    assert "using only the provided document context" in prompt
    assert "I don't know from this document." in prompt


def test_build_grounded_prompt_strips_question_whitespace():
    context = "Employees must carry their ID card."
    question = "  What must employees carry?  "

    prompt = llm.build_grounded_prompt(
        context,
        question,
    )

    assert "Question:\nWhat must employees carry?" in prompt


def test_build_grounded_prompt_rejects_empty_context():
    with pytest.raises(ValueError, match="Context cannot be empty"):
        llm.build_grounded_prompt(
            "",
            "What is the policy?",
        )


def test_build_grounded_prompt_rejects_whitespace_context():
    with pytest.raises(ValueError, match="Context cannot be empty"):
        llm.build_grounded_prompt(
            "   ",
            "What is the policy?",
        )


def test_build_grounded_prompt_rejects_empty_question():
    with pytest.raises(ValueError, match="Question cannot be empty"):
        llm.build_grounded_prompt(
            "The document contains information.",
            "",
        )


def test_build_grounded_prompt_rejects_whitespace_question():
    with pytest.raises(ValueError, match="Question cannot be empty"):
        llm.build_grounded_prompt(
            "The document contains information.",
            "   ",
        )


def test_create_gemini_client_rejects_missing_api_key(monkeypatch):
    monkeypatch.setattr(llm, "API_KEY", None)

    with pytest.raises(
        ValueError,
        match="GEMINI_API_KEY is not configured",
    ):
        llm.create_gemini_client()
