from rag.loader import load_pdfs
from rag.chunker import chunk_documents
from rag.embeddings import create_embeddings
from rag.vectorstore import reset_collection, add_chunks


DOCUMENTS_PATH = "documents"


def ingest_documents():

    print("Loading PDFs...")

    documents = load_pdfs(DOCUMENTS_PATH)

    print(f"Loaded {len(documents)} pages.")

    print("Creating chunks...")

    chunks = chunk_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    print("Creating embeddings...")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = create_embeddings(texts)

    print(f"Created {len(embeddings)} embeddings.")

    print("Opening ChromaDB (resetting collection with cosine space)...")

    collection = reset_collection()

    print("Adding data to ChromaDB...")

    add_chunks(
        collection,
        chunks,
        embeddings
    )

    print("Ingestion complete!")


if __name__ == "__main__":
    ingest_documents()