import pytest

from src.chat_history import (
    add_assistant_message,
    add_user_message,
    initialize_chat_history,
)


def test_initialize_chat_history_creates_empty_history():
    session_state = {}

    initialize_chat_history(session_state)

    assert session_state["chat_history"] == []


def test_initialize_chat_history_does_not_overwrite_existing_history():
    session_state = {
        "chat_history": [
            {
                "role": "user",
                "content": "Existing question",
            }
        ]
    }

    initialize_chat_history(session_state)

    assert len(session_state["chat_history"]) == 1
    assert session_state["chat_history"][0]["content"] == "Existing question"


def test_add_user_message_stores_message():
    session_state = {}

    add_user_message(
        session_state,
        "What is eDAR?",
    )

    assert session_state["chat_history"] == [
        {
            "role": "user",
            "content": "What is eDAR?",
        }
    ]


def test_add_assistant_message_stores_answer_and_sources():
    session_state = {}

    sources = [
        {
            "source_file": "test_document.pdf",
            "page_number": 5,
            "chunk_id": "page-005-chunk-001",
        }
    ]

    add_assistant_message(
        session_state,
        "eDAR is an electronic accident reporting system.",
        sources,
    )

    assert session_state["chat_history"] == [
        {
            "role": "assistant",
            "content": "eDAR is an electronic accident reporting system.",
            "sources": sources,
        }
    ]


def test_chat_history_preserves_message_order():
    session_state = {}

    add_user_message(session_state, "What is eDAR?")
    add_assistant_message(
        session_state,
        "eDAR is an electronic accident reporting system.",
    )
    add_user_message(session_state, "Why is it used?")
    add_assistant_message(
        session_state,
        "It is used for electronic accident reporting.",
    )

    history = session_state["chat_history"]

    assert len(history) == 4
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    assert history[2]["role"] == "user"
    assert history[3]["role"] == "assistant"


def test_assistant_message_without_sources_uses_empty_list():
    session_state = {}

    add_assistant_message(
        session_state,
        "This is an answer.",
    )

    assert session_state["chat_history"][0]["sources"] == []


def test_add_user_message_rejects_empty_content():
    session_state = {}

    with pytest.raises(ValueError, match="Message content cannot be empty"):
        add_user_message(session_state, "")


def test_add_assistant_message_rejects_empty_content():
    session_state = {}

    with pytest.raises(ValueError, match="Message content cannot be empty"):
        add_assistant_message(session_state, "   ")