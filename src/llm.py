import os

import streamlit as st
from dotenv import load_dotenv
from groq import Groq


load_dotenv()


MODEL_NAME = "openai/gpt-oss-120b"


def get_api_key() -> str | None:
    """Get the Groq API key from Streamlit secrets or environment variables."""
    try:
        return st.secrets["GROQ_API_KEY"]
    except st.errors.StreamlitSecretNotFoundError:
        return os.getenv("GROQ_API_KEY")
    except KeyError:
        return os.getenv("GROQ_API_KEY")


def create_groq_client():
    """Create and return a Groq client using the configured API key."""
    api_key = get_api_key()

    if not api_key:
        raise ValueError("GROQ_API_KEY is not configured.")

    return Groq(api_key=api_key)


def build_grounded_prompt(context: str, question: str) -> str:
    if not context or not context.strip():
        raise ValueError("Context cannot be empty.")

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    return (
        "Answer the question using ONLY information explicitly supported "
        "by the provided document context.\n\n"
        "Important rules:\n"
        "1. Do not use your general knowledge.\n"
        "2. Do not infer or guess information that is not explicitly supported by the context.\n"
        "3. The retrieved context may contain information unrelated to the question.\n"
        "4. If the context does not contain enough information to directly answer the question, respond exactly:\n"
        "I don't know from this document.\n"
        "5. Do not answer a different question using information that happens to appear in the context.\n\n"
        f"Document Context:\n{context}\n\n"
        f"Question:\n{question.strip()}\n"
    )


def generate_answer(context: str, question: str) -> str:
    """Generate an answer from Groq using the provided document context."""
    client = create_groq_client()
    prompt = build_grounded_prompt(context, question)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response.choices[0].message.content
