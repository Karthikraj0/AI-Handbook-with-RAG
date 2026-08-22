import pymupdf
from pathlib import Path


def load_pdfs(folder_path):
    documents = []

    folder = Path(folder_path)

    for pdf_file in folder.glob("*.pdf"):

        pdf = pymupdf.open(pdf_file)

        for page_number, page in enumerate(pdf, start=1):

            text = page.get_text("text").strip()

            if text:
                documents.append({
                    "text": text,
                    "source": pdf_file.name,
                    "page": page_number
                })

        pdf.close()

    return documents

class ImageOnlyPDFError(Exception):
    """Raised when an uploaded PDF contains only scanned images without selectable digital text."""
    pass


def load_pdf_from_bytes(file_bytes: bytes, filename: str):
    """
    Loads text page-by-page from an in-memory PDF byte stream (e.g. from Streamlit file_uploader).
    Raises ImageOnlyPDFError if the file has no extractable digital text or is purely image-based.
    """
    documents = []
    pdf = pymupdf.open(stream=file_bytes, filetype="pdf")
    total_pages = len(pdf)
    total_images = 0

    for page_number, page in enumerate(pdf, start=1):
        total_images += len(page.get_images())
        text = page.get_text("text").strip()
        if text:
            documents.append({
                "text": text,
                "source": filename,
                "page": page_number
            })

    pdf.close()

    if total_pages > 0 and not documents:
        if total_images > 0:
            raise ImageOnlyPDFError(
                f"'{filename}' contains {total_pages} scanned page(s) and {total_images} image(s), but no selectable digital text."
            )
        else:
            raise ImageOnlyPDFError(
                f"'{filename}' contains no readable text or is an empty document."
            )

    # Check for near-empty text while containing images (e.g. minimal OCR artifacts < 30 chars total)
    total_chars = sum(len(d["text"]) for d in documents)
    if total_chars < 30 and total_images > 0:
        raise ImageOnlyPDFError(
            f"'{filename}' appears to be a scanned image document without searchable text."
        )

    return documents


if __name__ == "__main__":
    documents = load_pdfs("documents")
    print("Number of documents:", len(documents))

