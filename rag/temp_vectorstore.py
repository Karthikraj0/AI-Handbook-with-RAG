import logging
import chromadb
from rag.loader import load_pdf_from_bytes, ImageOnlyPDFError
from rag.chunker import chunk_documents
from rag.embeddings import create_embeddings, create_embedding

logger = logging.getLogger(__name__)

# In-memory ephemeral Chroma client (zero disk persistence, isolated per runtime session)
_ephemeral_client = chromadb.EphemeralClient()


def _get_collection_name(session_id: str) -> str:
    # Ensure safe alphanumeric collection name
    safe_id = "".join(c for c in session_id if c.isalnum() or c in ("_", "-"))
    return f"temp_session_{safe_id}"


def index_temp_pdf(session_id: str, file_bytes: bytes, filename: str) -> dict:
    """
    Extracts, chunks, embeds, and stores an uploaded PDF into an in-memory ChromaDB collection.
    Returns metadata stats: {'filename': str, 'pages': int, 'chunks': int}.
    """
    collection_name = _get_collection_name(session_id)

    # 1. Extract pages from in-memory byte stream
    documents = load_pdf_from_bytes(file_bytes, filename)
    if not documents:
        raise ValueError("No extractable text found in the uploaded PDF. It may be a scanned or image-only file.")

    # 2. Chunk text with 300-word chunk size for comprehensive document coverage
    chunks = chunk_documents(documents, chunk_size=300, overlap=50)
    if not chunks:
        raise ValueError("Could not generate valid text chunks from the uploaded PDF.")

    # 3. Create vector embeddings
    texts = [chunk["text"] for chunk in chunks]
    embeddings = create_embeddings(texts)

    # 4. Clean existing session collection if present
    try:
        _ephemeral_client.delete_collection(name=collection_name)
    except Exception:
        pass

    # 5. Create fresh in-memory collection with Cosine similarity
    collection = _ephemeral_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    # 6. Add chunks into collection
    ids = [f"{filename}_p{chunk['page']}_{i}" for i, chunk in enumerate(chunks)]
    metadatas = [{"source": chunk["source"], "page": chunk["page"]} for chunk in chunks]

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings
    )

    logger.info(f"[TempVectorStore] Indexed '{filename}' into '{collection_name}' ({len(documents)} pages, {len(chunks)} chunks)")

    return {
        "filename": filename,
        "pages": len(documents),
        "chunks": len(chunks)
    }


def retrieve_temp_documents(session_id: str, query: str, n_results: int = 3, reformulated_query: str = None) -> dict:
    """
    Retrieves nearest matching chunks from the session's temporary in-memory collection.
    """
    collection_name = _get_collection_name(session_id)
    try:
        collection = _ephemeral_client.get_collection(name=collection_name)
    except Exception:
        logger.warning(f"[TempVectorStore] Temporary collection '{collection_name}' not found.")
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    search_query = reformulated_query if (reformulated_query and reformulated_query.strip()) else query
    query_embedding = create_embedding(search_query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results


def clear_temp_session(session_id: str):
    """
    Discards the temporary in-memory collection for the given session.
    """
    collection_name = _get_collection_name(session_id)
    try:
        _ephemeral_client.delete_collection(name=collection_name)
        logger.info(f"[TempVectorStore] Cleared temporary collection '{collection_name}'")
    except Exception:
        pass


def has_temp_session(session_id: str) -> bool:
    """
    Returns True if an active temporary collection exists for the session.
    """
    collection_name = _get_collection_name(session_id)
    try:
        col = _ephemeral_client.get_collection(name=collection_name)
        return col.count() > 0
    except Exception:
        return False
