from dataclasses import dataclass


@dataclass
class DocumentChunk:
    """A PDF text chunk together with its source metadata."""

    text: str
    source_file: str
    page_number: int
    chunk_id: str