from pathlib import Path

from pypdf import PdfReader
from docx import Document


def load_txt(file_path: str) -> list[dict]:
    """Load a TXT file."""

    path = Path(file_path)

    text = path.read_text(
        encoding="utf-8"
    )

    return [
        {
            "text": text,
            "metadata": {
                "source": path.name,
                "file_type": "txt",
            }
        }
    ]


def load_pdf(file_path: str) -> list[dict]:
    """Load a PDF file page by page."""

    path = Path(file_path)

    reader = PdfReader(file_path)

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text() or ""

        if text.strip():

            documents.append(
                {
                    "text": text,
                    "metadata": {
                        "source": path.name,
                        "file_type": "pdf",
                        "page": page_number,
                    }
                }
            )

    return documents


def load_docx(file_path: str) -> list[dict]:
    """Load a DOCX file."""

    path = Path(file_path)

    document = Document(file_path)

    text = "\n".join(
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    )

    return [
        {
            "text": text,
            "metadata": {
                "source": path.name,
                "file_type": "docx",
            }
        }
    ]


def load_document(file_path: str) -> list[dict]:
    """Automatically select the correct loader."""

    extension = Path(file_path).suffix.lower()

    if extension == ".txt":
        return load_txt(file_path)

    elif extension == ".pdf":
        return load_pdf(file_path)

    elif extension == ".docx":
        return load_docx(file_path)

    else:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )