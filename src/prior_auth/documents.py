"""Text extraction from uploaded supporting documents."""

from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class DocumentError(ValueError):
    """The upload is not a readable PDF, or it has no text layer."""


def extract_pdf_text(data: bytes, max_chars: int) -> str:
    if not data.startswith(b"%PDF"):
        raise DocumentError("The supporting document must be a PDF.")
    try:
        reader = PdfReader(BytesIO(data))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    except (PdfReadError, ValueError, KeyError) as exc:
        raise DocumentError(f"The PDF could not be read: {exc}") from exc
    text = text.strip()
    if not text:
        # Scanned PDFs have no text layer. OCR is listed as a next step in the README.
        raise DocumentError("The PDF has no extractable text (scanned documents are not supported yet).")
    return text[:max_chars]
