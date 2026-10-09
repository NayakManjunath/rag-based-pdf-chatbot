
from src import llm


def test_generate_answer_returns_exact_fallback_when_model_returns_fallback(
    monkeypatch,
):
    expected_answer = "I don't know from this document."
    captured = {}

    class FakeMessage:
        content = expected_answer

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, model, messages):
            captured["model"] = model
            captured["prompt"] = messages[0]["content"]
            return FakeResponse()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr(llm, "create_groq_client", lambda: FakeClient())

    answer = llm.generate_answer(
        context="The document describes an electronic reporting system.",
        question="What is the capital of France?",
    )

    assert answer == expected_answer
    assert captured["model"] == llm.MODEL_NAME
    assert expected_answer in captured["prompt"]
    assert "ONLY information explicitly supported" in captured["prompt"]