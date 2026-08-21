from rag.vectorstore import get_collection
from rag.embeddings import create_embedding


RELEVANCE_THRESHOLD = 0.47


def retrieve_documents(query, n_results=3):

    collection = get_collection()

    # Convert question into embedding
    query_embedding = create_embedding(query)

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    # Get the distance of the best result
    best_distance = results["distances"][0][0]

    # Reject if even the best result is not relevant enough
    if best_distance > RELEVANCE_THRESHOLD:
        return None

    return results