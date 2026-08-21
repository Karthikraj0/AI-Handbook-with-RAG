from rag.vectorstore import get_collection
from rag.embeddings import create_embedding


def retrieve_documents(query, n_results=3):

    collection = get_collection()

    query_embedding = create_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    return results