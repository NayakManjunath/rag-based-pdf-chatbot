import os

from dotenv import load_dotenv
from google import genai


load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-3.5-flash"


def create_gemini_client():
    """Create and return a Gemini client using the configured API key."""
    if not API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured.")

    return genai.Client(api_key=API_KEY)


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
    """Generate an answer from Gemini using the provided document context."""
    client = create_gemini_client()
    prompt = build_grounded_prompt(context, question)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    return response.text