from src.query_resolver import resolve_question


def test_follow_up_question_produces_resolved_retrieval_query(monkeypatch):
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

    monkeypatch.setattr(
        "src.query_resolver.create_groq_client",
        lambda: _fake_groq_client("Who uses eDAR?"),
    )

    resolved_question = resolve_question(
        history,
        "Who uses it?",
    )

    assert resolved_question == "Who uses eDAR?"


def _fake_groq_client(resolved_question):
    class FakeMessage:
        content = resolved_question

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

    return FakeClient()