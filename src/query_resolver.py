from src.llm import MODEL_NAME, create_groq_client


def build_follow_up_prompt(
    chat_history: list[dict],
    question: str,
) -> str:
    """
    Build a prompt that converts a conversational question into a
    standalone retrieval question.

    The previous conversation is used only to resolve references such
    as "it", "they", "this", or "that". The model must not answer the
    question or introduce unrelated information.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not chat_history:
        raise ValueError("Chat history cannot be empty.")

    previous_messages = []

    for message in chat_history:
        role = message.get("role")
        content = message.get("content")

        if not role or not content:
            continue

        if role == "user":
            previous_messages.append(
                f"User: {content.strip()}"
            )
        elif role == "assistant":
            previous_messages.append(
                f"Assistant: {content.strip()}"
            )

    if not previous_messages:
        raise ValueError("Chat history does not contain usable messages.")

    conversation = "\n".join(previous_messages)

    return (
        "Rewrite the current question as a standalone question for "
        "retrieving information from a document.\n\n"
        "Rules:\n"
        "1. Use the previous conversation only to resolve references "
        "such as it, they, this, that, he, she, or similar references.\n"
        "2. Do not answer the question.\n"
        "3. Do not add facts that are not present in the conversation.\n"
        "4. If the current question is already standalone, return it "
        "unchanged.\n"
        "5. Return only the rewritten question.\n\n"
        f"Previous conversation:\n{conversation}\n\n"
        f"Current question:\n{question.strip()}\n"
    )


def resolve_question(
    chat_history: list[dict],
    question: str,
) -> str:
    """
    Resolve a conversational question into a standalone retrieval query.

    If there is no previous conversation, the original question is
    returned unchanged and no LLM call is required.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not chat_history:
        return question.strip()

    prompt = build_follow_up_prompt(
        chat_history=chat_history,
        question=question,
    )

    client = create_groq_client()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    resolved_question = response.choices[0].message.content

    if not resolved_question or not resolved_question.strip():
        raise ValueError("Question resolution returned an empty response.")

    return resolved_question.strip()