import chromadb


CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "company_policies"


_collection = None


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(
            path=CHROMA_PATH
        )
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


def reset_collection():
    global _collection
    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    _collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    return _collection



def add_chunks(collection, chunks, embeddings):

    ids = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):

        ids.append(f"chunk_{i}")

        documents.append(chunk["text"])

        metadatas.append({
            "source": chunk["source"],
            "page": chunk["page"]
        })

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )