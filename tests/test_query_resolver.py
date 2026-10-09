import pytest

import src.query_resolver as query_resolver


def test_resolve_question_without_history_returns_original_question():
    question = "What is RAG?"

    result = query_resolver.resolve_question(
        chat_history=[],
        question=question,
    )

    assert result == "What is RAG?"


def test_resolve_question_strips_question_whitespace_without_history():
    question = "   What is RAG?   "

    result = query_resolver.resolve_question(
        chat_history=[],
        question=question,
    )

    assert result == "What is RAG?"


def test_resolve_question_rejects_empty_question():
    with pytest.raises(ValueError, match="Question cannot be empty"):
        query_resolver.resolve_question(
            chat_history=[],
            question="",
        )


def test_build_follow_up_prompt_contains_conversation_and_question():
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

    prompt = query_resolver.build_follow_up_prompt(
        chat_history=history,
        question="Who uses it?",
    )

    assert "What is eDAR?" in prompt
    assert "eDAR is an electronic accident reporting system." in prompt
    assert "Who uses it?" in prompt


def test_build_follow_up_prompt_requires_question():
    history = [
        {
            "role": "user",
            "content": "What is eDAR?",
        }
    ]

    with pytest.raises(ValueError, match="Question cannot be empty"):
        query_resolver.build_follow_up_prompt(
            chat_history=history,
            question="",
        )


def test_build_follow_up_prompt_requires_history():
    with pytest.raises(ValueError, match="Chat history cannot be empty"):
        query_resolver.build_follow_up_prompt(
            chat_history=[],
            question="Who uses it?",
        )


def test_resolve_question_uses_groq_to_resolve_follow_up(monkeypatch):
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

    class FakeMessage:
        content = "Who uses eDAR?"

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, model, messages):
            assert model == query_resolver.MODEL_NAME
            assert len(messages) == 1
            assert "What is eDAR?" in messages[0]["content"]
            assert "Who uses it?" in messages[0]["content"]

            return FakeResponse()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr(
        query_resolver,
        "create_groq_client",
        lambda: FakeClient(),
    )

    result = query_resolver.resolve_question(
        chat_history=history,
        question="Who uses it?",
    )

    assert result == "Who uses eDAR?"


def test_resolve_question_rejects_empty_model_response(monkeypatch):
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

    class FakeMessage:
        content = ""

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, model, messages):
            return FakeResponse()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr(
        query_resolver,
        "create_groq_client",
        lambda: FakeClient(),
    )

    with pytest.raises(
        ValueError,
        match="Question resolution returned an empty response",
    ):
        query_resolver.resolve_question(
            chat_history=history,
            question="Who uses it?",
        )