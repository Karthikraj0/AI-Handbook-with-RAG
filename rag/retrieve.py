import logging
from rag.vectorstore import get_collection
from rag.embeddings import create_embedding
from rag.query_rewriter import reformulate_query

logger = logging.getLogger(__name__)

RELEVANCE_THRESHOLD = 0.47


def retrieve_documents(query, n_results=5, reformulated_query=None):
    """
    Retrieves documents from ChromaDB based on the query.
    If `reformulated_query` is provided, it is used for embedding and vector search.
    Otherwise, `reformulate_query(query)` is called.
    """
    collection = get_collection()

    # Obtain search query (using provided reformulated query or reformulate once)
    search_query = reformulated_query if reformulated_query is not None else reformulate_query(query)

    # Convert search query into embedding
    query_embedding = create_embedding(search_query)

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    if not results or not results.get("distances") or not results["distances"][0]:
        return None

    # Get the distance of the best result
    best_distance = results["distances"][0][0]

    # Reject if even the best result is not relevant enough
    if best_distance > RELEVANCE_THRESHOLD:
        return None

    return results