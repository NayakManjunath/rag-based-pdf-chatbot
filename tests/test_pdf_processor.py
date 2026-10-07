from pathlib import Path

import pytest

from src.pdf_processor import (
    create_chunks,
    create_document_chunks,
    extract_pages_from_pdf,
    extract_text_from_pdf,
    save_uploaded_pdf,
)


def test_save_uploaded_pdf_copies_pdf_to_upload_directory(tmp_path, monkeypatch):
    source_pdf = tmp_path / "sample.pdf"
    source_pdf.write_bytes(b"%PDF-test-content")

    upload_dir = tmp_path / "uploads"
    monkeypatch.setattr(
        "src.pdf_processor.UPLOAD_DIR",
        upload_dir,
    )

    saved_path = save_uploaded_pdf(str(source_pdf))

    assert saved_path == upload_dir / "sample.pdf"
    assert saved_path.exists()
    assert saved_path.read_bytes() == b"%PDF-test-content"


def test_save_uploaded_pdf_rejects_missing_file(tmp_path):
    missing_pdf = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        save_uploaded_pdf(str(missing_pdf))


def test_save_uploaded_pdf_rejects_non_pdf_file(tmp_path):
    text_file = tmp_path / "sample.txt"
    text_file.write_text("test content", encoding="utf-8")

    with pytest.raises(ValueError, match="Only PDF files are supported"):
        save_uploaded_pdf(str(text_file))


def test_extract_text_from_pdf_rejects_missing_file(tmp_path):
    missing_pdf = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        extract_text_from_pdf(missing_pdf)


def test_extract_text_from_pdf_rejects_non_pdf_file(tmp_path):
    text_file = tmp_path / "sample.txt"
    text_file.write_text("test content", encoding="utf-8")

    with pytest.raises(ValueError, match="Only PDF files are supported"):
        extract_text_from_pdf(text_file)


def test_create_chunks_splits_text_into_fixed_size_chunks():
    text = "A" * 1750

    chunks = create_chunks(text, chunk_size=800)

    assert len(chunks) == 3
    assert [len(chunk) for chunk in chunks] == [800, 800, 150]


def test_create_chunks_returns_empty_list_for_empty_text():
    assert create_chunks("") == []


def test_create_chunks_rejects_invalid_chunk_size():
    with pytest.raises(ValueError, match="chunk_size must be greater than zero"):
        create_chunks("sample text", chunk_size=0)

def test_extract_pages_from_pdf_preserves_page_numbers(tmp_path):
    from pypdf import PdfWriter

    pdf_path = tmp_path / "sample.pdf"

    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_blank_page(width=72, height=72)

    with pdf_path.open("wb") as file:
        writer.write(file)

    pages = extract_pages_from_pdf(pdf_path)

    assert len(pages) == 2
    assert pages[0]["page_number"] == 1
    assert pages[1]["page_number"] == 2

def test_create_document_chunks_preserves_source_filename():
    pages = [
        {
            "page_number": 1,
            "text": "This is page one content.",
        }
    ]

    chunks = create_document_chunks(
        pages,
        source_file="sample.pdf",
        chunk_size=800,
    )

    assert len(chunks) == 1
    assert chunks[0].source_file == "sample.pdf"

def test_create_document_chunks_generates_unique_chunk_ids():
    pages = [
        {
            "page_number": 1,
            "text": "A" * 1700,
        }
    ]

    chunks = create_document_chunks(
        pages,
        source_file="sample.pdf",
        chunk_size=800,
    )

    chunk_ids = [
        chunk.chunk_id
        for chunk in chunks
    ]

    assert len(chunk_ids) == len(set(chunk_ids))


def test_create_document_chunks_preserves_chunk_text():
    text = "This is the original PDF text."

    pages = [
        {
            "page_number": 1,
            "text": text,
        }
    ]

    chunks = create_document_chunks(
        pages,
        source_file="sample.pdf",
        chunk_size=800,
    )

    assert chunks[0].text == text