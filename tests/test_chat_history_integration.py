import pytest

from src.chat_history import (
    add_assistant_message,
    add_user_message,
    get_pdf_id,
    reset_chat_history_for_new_pdf,
)


def test_pdf_id_is_deterministic():
    pdf_bytes = b"sample pdf content"

    first_id = get_pdf_id(pdf_bytes)
    second_id = get_pdf_id(pdf_bytes)

    assert first_id == second_id


def test_different_pdf_content_produces_different_pdf_id():
    first_id = get_pdf_id(b"first pdf")
    second_id = get_pdf_id(b"second pdf")

    assert first_id != second_id


def test_same_pdf_preserves_existing_chat_history():
    session_state = {}

    pdf_id = get_pdf_id(b"same pdf")

    reset_chat_history_for_new_pdf(
        session_state,
        pdf_id,
    )

    add_user_message(
        session_state,
        "What is RAG?",
    )

    add_assistant_message(
        session_state,
        "RAG is Retrieval-Augmented Generation.",
    )

    reset_chat_history_for_new_pdf(
        session_state,
        pdf_id,
    )

    assert len(session_state["chat_history"]) == 2
    assert session_state["chat_history"][0]["role"] == "user"
    assert session_state["chat_history"][1]["role"] == "assistant"


def test_new_pdf_clears_existing_chat_history():
    session_state = {}

    first_pdf_id = get_pdf_id(b"first pdf")
    second_pdf_id = get_pdf_id(b"second pdf")

    reset_chat_history_for_new_pdf(
        session_state,
        first_pdf_id,
    )

    add_user_message(
        session_state,
        "What is RAG?",
    )

    add_assistant_message(
        session_state,
        "RAG is Retrieval-Augmented Generation.",
    )

    reset_chat_history_for_new_pdf(
        session_state,
        second_pdf_id,
    )

    assert session_state["current_pdf_id"] == second_pdf_id
    assert session_state["chat_history"] == []


def test_conversation_remains_in_order_after_multiple_reruns():
    session_state = {}

    pdf_id = get_pdf_id(b"same pdf")

    reset_chat_history_for_new_pdf(
        session_state,
        pdf_id,
    )

    add_user_message(
        session_state,
        "Question 1",
    )

    add_assistant_message(
        session_state,
        "Answer 1",
    )

    reset_chat_history_for_new_pdf(
        session_state,
        pdf_id,
    )

    add_user_message(
        session_state,
        "Question 2",
    )

    add_assistant_message(
        session_state,
        "Answer 2",
    )

    reset_chat_history_for_new_pdf(
        session_state,
        pdf_id,
    )

    add_user_message(
        session_state,
        "Question 3",
    )

    add_assistant_message(
        session_state,
        "Answer 3",
    )

    history = session_state["chat_history"]

    assert len(history) == 6

    assert history[0]["content"] == "Question 1"
    assert history[1]["content"] == "Answer 1"
    assert history[2]["content"] == "Question 2"
    assert history[3]["content"] == "Answer 2"
    assert history[4]["content"] == "Question 3"
    assert history[5]["content"] == "Answer 3"


def test_pdf_id_rejects_empty_content():
    with pytest.raises(
        ValueError,
        match="PDF content cannot be empty",
    ):
        get_pdf_id(b"")


def test_pdf_id_rejects_empty_identifier():
    session_state = {}

    with pytest.raises(
        ValueError,
        match="PDF ID cannot be empty",
    ):
        reset_chat_history_for_new_pdf(
            session_state,
            "",
        )