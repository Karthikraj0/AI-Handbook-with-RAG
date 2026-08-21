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

if __name__ == "__main__":
    documents = load_pdfs("documents")
    print("Number of documents:", len(documents))

