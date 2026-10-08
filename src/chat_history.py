
import hashlib


def initialize_chat_history(session_state) -> None:
    """Initialize chat history if it does not already exist."""
    if "chat_history" not in session_state:
        session_state["chat_history"] = []


def add_user_message(session_state, content: str) -> None:
    """Add a user message to the current conversation."""
    if not content or not content.strip():
        raise ValueError("Message content cannot be empty.")

    initialize_chat_history(session_state)

    session_state["chat_history"].append(
        {
            "role": "user",
            "content": content.strip(),
        }
    )


def add_assistant_message(
    session_state,
    content: str,
    sources: list[dict] | None = None,
) -> None:
    """Add an assistant response and its source evidence."""
    if not content or not content.strip():
        raise ValueError("Message content cannot be empty.")

    initialize_chat_history(session_state)

    session_state["chat_history"].append(
        {
            "role": "assistant",
            "content": content.strip(),
            "sources": sources or [],
        }
    )

def get_pdf_id(pdf_bytes: bytes) -> str:
    """Return a deterministic identifier for uploaded PDF content."""
    if not pdf_bytes:
        raise ValueError("PDF content cannot be empty.")

    return hashlib.sha256(pdf_bytes).hexdigest()


def reset_chat_history_for_new_pdf(
    session_state,
    pdf_id: str,
) -> None:
    """
    Start a fresh conversation when a different PDF is uploaded.

    If the same PDF is uploaded again, the existing conversation
    remains unchanged.
    """
    if not pdf_id or not pdf_id.strip():
        raise ValueError("PDF ID cannot be empty.")

    initialize_chat_history(session_state)

    if session_state.get("current_pdf_id") != pdf_id:
        session_state["current_pdf_id"] = pdf_id
        session_state["chat_history"] = []