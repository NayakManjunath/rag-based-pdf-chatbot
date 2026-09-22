from pathlib import Path

from pypdf import PdfWriter
from src import pdf_processor

from src.embeddings import generate_embeddings
from src.llm import build_grounded_prompt
from src.pdf_processor import create_chunks
from src.rag_pipeline import build_context
from src.retriever import embed_question, retrieve_relevant_chunks
from src.vector_store import create_faiss_index


def create_test_pdf(pdf_path: Path):
    """Create a small PDF containing deterministic test information."""
    writer = PdfWriter()

    writer.add_blank_page(width=612, height=792)

    with pdf_path.open("wb") as pdf_file:
        writer.write(pdf_file)


def test_end_to_end_rag_pipeline(tmp_path, monkeypatch):
    """Validate the complete RAG pipeline from PDF to grounded prompt."""
    pdf_path = tmp_path / "test_document.pdf"

    create_test_pdf(pdf_path)

    # Use deterministic text for the test while still exercising
    # the application's PDF-processing and RAG stages.
    test_text = (
        "FAISS is used for efficient similarity search of embeddings. "
        "The RAG chatbot retrieves relevant document chunks before "
        "generating an answer."
    )

    monkeypatch.setattr(
        "src.pdf_processor.extract_text_from_pdf",
        lambda _: test_text,
    )

    monkeypatch.setattr(
        "src.pdf_processor.extract_text_from_pdf",
        lambda _: test_text,
    )

    

    extracted_text = pdf_processor.extract_text_from_pdf(pdf_path)
    chunks = create_chunks(
        extracted_text,
        chunk_size=800,
    )

    assert chunks

    embeddings = generate_embeddings(chunks)

    index = create_faiss_index(embeddings)

    question = "What is FAISS used for?"

    question_embedding = embed_question(question)

    retrieved_chunks = retrieve_relevant_chunks(
        index,
        chunks,
        question_embedding,
        k=1,
    )

    assert retrieved_chunks
    assert "FAISS" in retrieved_chunks[0]

    context = build_context(retrieved_chunks)

    assert "FAISS" in context
    assert "similarity search" in context

    prompt = build_grounded_prompt(
        context,
        question,
    )

    assert "FAISS" in prompt
    assert question in prompt
    assert "Answer the question using only the provided document context." in prompt
