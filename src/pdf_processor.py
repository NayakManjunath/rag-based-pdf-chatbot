from pathlib import Path

from pypdf import PdfReader

from src.models import DocumentChunk


UPLOAD_DIR = Path("data/uploads")


def save_uploaded_pdf(file_path: str) -> Path:
    """Validate and save a PDF file to the project's upload directory."""
    source_path = Path(file_path)

    if not source_path.exists():
        raise FileNotFoundError(f"PDF file not found: {source_path}")

    if source_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    destination_path = UPLOAD_DIR / source_path.name
    destination_path.write_bytes(source_path.read_bytes())

    return destination_path


def extract_text_from_pdf(file_path: str | Path) -> str:
    """Extract text from all pages of a PDF."""
    pdf_path = Path(file_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    reader = PdfReader(pdf_path)

    page_text = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            page_text.append(text)

    return "\n".join(page_text)


def extract_pages_from_pdf(file_path: str | Path) -> list[dict]:
    """Extract text from each PDF page while preserving page numbers."""
    pdf_path = Path(file_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        pages.append(
            {
                "page_number": page_number,
                "text": text,
            }
        )

    return pages


def create_chunks(
    text: str,
    chunk_size: int = 800,
) -> list[str]:
    """Split document text into fixed-size character chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if not text:
        return []

    return [
        text[start:start + chunk_size]
        for start in range(0, len(text), chunk_size)
    ]


def create_document_chunks(
    pages: list[dict],
    source_file: str,
    chunk_size: int = 800,
) -> list[DocumentChunk]:
    """Create page-aware chunks with source metadata."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if not source_file or not source_file.strip():
        raise ValueError("source_file cannot be empty.")

    if not pages:
        return []

    chunks = []
    chunk_counter = 1

    for page in pages:
        page_number = page["page_number"]
        text = page["text"]

        if not text:
            continue

        for start in range(0, len(text), chunk_size):
            chunk_text = text[start:start + chunk_size]

            chunk_id = (
                f"page-{page_number:03d}-"
                f"chunk-{chunk_counter:03d}"
            )

            chunks.append(
                DocumentChunk(
                    text=chunk_text,
                    source_file=source_file,
                    page_number=page_number,
                    chunk_id=chunk_id,
                )
            )

            chunk_counter += 1

    return chunks