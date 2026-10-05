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
    """Build a prompt that restricts the answer to the PDF context."""
    if not context or not context.strip():
        raise ValueError("Context cannot be empty.")

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    return f"""Answer the question using only the provided document context.

If the answer is not available in the document context, respond exactly:
I don't know from this document.

Document Context:
{context}

Question:
{question.strip()}
"""


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
