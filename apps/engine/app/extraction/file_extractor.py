from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pymupdf
from docx import Document


SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".xml",
    ".html",
    ".css",
    ".csv",
    ".sql",
    ".rs",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".pdf",
    ".docx",
}


# Files at or below this size can safely be extracted as a whole.
FULL_TEXT_MAX_BYTES = 5 * 1024 * 1024

# Files above this size are normally represented using metadata only.
# PDF/DOCX files are an exception and can still use chunked extraction.
METADATA_ONLY_MAX_BYTES = 50 * 1024 * 1024

# Maximum approximate amount of text accumulated in one chunk.
CHUNK_SIZE_CHARS = 100_000


def get_indexing_mode(
    path: Path,
    size_bytes: int,
) -> str:
    """
    Decide how RecallX should index a file.

    Returns:
        "full":
            Extract the complete file into memory.

        "chunked":
            Read/process the file incrementally.

        "metadata_only":
            Do not read the file contents. Store lightweight metadata.
    """
    extension = path.suffix.lower()

    # Small files are safe to process completely.
    if size_bytes <= FULL_TEXT_MAX_BYTES:
        return "full"

    # Large PDFs and DOCX files can still be processed incrementally.
    if extension in {".pdf", ".docx"}:
        return "chunked"

    # Medium-sized text/code files can be processed incrementally.
    if size_bytes <= METADATA_ONLY_MAX_BYTES:
        return "chunked"

    # Very large files are represented through metadata only.
    return "metadata_only"


def extract_text(path: Path) -> str:
    """
    Extract the complete text from a supported file.

    This function should only be used when the selected indexing
    mode is "full".
    """
    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        return ""

    try:
        if extension == ".pdf":
            return _extract_pdf(path)

        if extension == ".docx":
            return _extract_docx(path)

        return _extract_text_file(path)

    except Exception as error:
        print(
            f"RecallX: failed to extract "
            f"{path}: {type(error).__name__}: {error}"
        )
        return ""


def _extract_text_file(path: Path) -> str:
    """
    Extract a normal text/code file.
    """
    return path.read_text(
        encoding="utf-8",
        errors="ignore",
    )


def _extract_pdf(path: Path) -> str:
    """
    Extract all text from a PDF.

    Used only for files selected for full extraction.
    """
    pages: list[str] = []

    with pymupdf.open(path) as document:
        for page in document:
            text = page.get_text()

            if text.strip():
                pages.append(text)

    return "\n\n".join(pages)


def _extract_docx(path: Path) -> str:
    """
    Extract paragraph text from a DOCX document.

    Used only for files selected for full extraction.
    """
    document = Document(path)

    paragraphs = [
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]

    return "\n".join(paragraphs)


def iter_text_chunks(
    path: Path,
    chunk_size: int = CHUNK_SIZE_CHARS,
) -> Iterator[str]:
    """
    Yield file content incrementally.

    The complete file is not loaded into memory at once.

    This is used for files selected for "chunked" indexing.
    """
    extension = path.suffix.lower()

    if extension in {
        ".txt",
        ".md",
        ".py",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".xml",
        ".html",
        ".css",
        ".csv",
        ".sql",
        ".rs",
        ".java",
        ".cpp",
        ".c",
        ".h",
        ".hpp",
    }:
        yield from _iter_text_file_chunks(
            path,
            chunk_size,
        )
        return

    if extension == ".pdf":
        yield from _iter_pdf_chunks(
            path,
            chunk_size,
        )
        return

    if extension == ".docx":
        yield from _iter_docx_chunks(
            path,
            chunk_size,
        )


def _iter_text_file_chunks(
    path: Path,
    chunk_size: int,
) -> Iterator[str]:
    """
    Incrementally read a text/code file.

    Lines are accumulated until approximately chunk_size characters
    have been collected.
    """
    buffer: list[str] = []
    buffer_size = 0

    with path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as file:
        for line in file:
            buffer.append(line)
            buffer_size += len(line)

            if buffer_size >= chunk_size:
                chunk = "".join(buffer).strip()

                if chunk:
                    yield chunk

                buffer.clear()
                buffer_size = 0

    if buffer:
        chunk = "".join(buffer).strip()

        if chunk:
            yield chunk


def _iter_pdf_chunks(
    path: Path,
    chunk_size: int,
) -> Iterator[str]:
    """
    Extract PDF text page-by-page and yield bounded chunks.

    Page numbers are preserved so retrieved context can identify
    where the text originated.
    """
    buffer: list[str] = []
    buffer_size = 0

    with pymupdf.open(path) as document:
        for page_number, page in enumerate(
            document,
            start=1,
        ):
            text = page.get_text().strip()

            if not text:
                continue

            page_text = (
                f"[Page {page_number}]\n"
                f"{text}"
            )

            buffer.append(page_text)
            buffer_size += len(page_text)

            if buffer_size >= chunk_size:
                chunk = "\n\n".join(buffer).strip()

                if chunk:
                    yield chunk

                buffer.clear()
                buffer_size = 0

    if buffer:
        chunk = "\n\n".join(buffer).strip()

        if chunk:
            yield chunk


def _iter_docx_chunks(
    path: Path,
    chunk_size: int,
) -> Iterator[str]:
    """
    Extract DOCX paragraphs incrementally.

    Paragraph boundaries are preserved while constructing chunks.
    """
    document = Document(path)

    buffer: list[str] = []
    buffer_size = 0

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()

        if not text:
            continue

        buffer.append(text)
        buffer_size += len(text)

        if buffer_size >= chunk_size:
            chunk = "\n".join(buffer).strip()

            if chunk:
                yield chunk

            buffer.clear()
            buffer_size = 0

    if buffer:
        chunk = "\n".join(buffer).strip()

        if chunk:
            yield chunk


def build_metadata_context(
    path: Path,
    size_bytes: int,
) -> str:
    """
    Build a lightweight memory representation for a huge file.

    The original file contents are not read.
    """
    extension = path.suffix.lower().lstrip(".")

    file_type = extension.upper() if extension else "FILE"

    return (
        f"Large {file_type} file named {path.name}. "
        f"Located at {path.parent}. "
        f"File size is {_format_size(size_bytes)}. "
        f"RecallX stored metadata only for this file."
    )


def _format_size(size_bytes: int) -> str:
    """
    Convert a byte count into a human-readable size.
    """
    size = float(size_bytes)

    units = (
        "B",
        "KB",
        "MB",
        "GB",
        "TB",
    )

    for unit in units:
        if size < 1024:
            return f"{size:.1f} {unit}"

        size /= 1024

    return f"{size:.1f} PB"