import logging
from rag.query_rewriter import reformulate_query
from rag.retrieve import retrieve_documents
from rag.generator import generate_answer, generate_answer_stream
from rag.temp_vectorstore import retrieve_temp_documents, has_temp_session

logger = logging.getLogger(__name__)

DISTANCE_THRESHOLD = 0.47
NO_POLICY_FOUND_MSG = "I couldn't find that information in the company policies."
NO_TEMP_FOUND_MSG = "I couldn't find that information in the uploaded document."


def _filter_results_by_threshold(results, threshold=DISTANCE_THRESHOLD):
    if results is None or not results.get("documents") or not results["documents"][0]:
        return [], []

    docs = results["documents"][0]
    metas = results["metadatas"][0] if results.get("metadatas") and results["metadatas"][0] else [{}] * len(docs)
    distances = results["distances"][0] if results.get("distances") and results["distances"][0] else [0.0] * len(docs)

    # If the top match exceeds the distance threshold, consider the question irrelevant
    if distances and distances[0] > threshold:
        return [], []

    filtered_docs = []
    filtered_metas = []

    for doc, meta, dist in zip(docs, metas, distances):
        if dist <= threshold:
            filtered_docs.append(doc)
            filtered_metas.append(meta)

    return filtered_docs, filtered_metas


def ask_question(query, temp_session_id: str = None):
    # Step 1: Reformulate query for semantic search (with safe fallback)
    reformulated_query = reformulate_query(query)

    # Step 2: Retrieve relevant policy chunks (Temporary session vs Permanent DB)
    if temp_session_id and has_temp_session(temp_session_id):
        results = retrieve_temp_documents(
            session_id=temp_session_id,
            query=query,
            n_results=5,
            reformulated_query=reformulated_query
        )
        fallback_msg = NO_TEMP_FOUND_MSG
    else:
        results = retrieve_documents(
            query,
            n_results=5,
            reformulated_query=reformulated_query
        )
        fallback_msg = NO_POLICY_FOUND_MSG

    documents, sources = _filter_results_by_threshold(results, DISTANCE_THRESHOLD)

    # Log retrieval metrics for observability
    top_dist = results["distances"][0][0] if results and results.get("distances") and results["distances"][0] else None
    logger.info(f"[RAG Pipeline] TempSession: {temp_session_id} | Original: '{query}' | Reformulated: '{reformulated_query}' | Top Distance: {top_dist}")

    # No relevant context found within threshold
    if not documents:
        return {
            "answer": fallback_msg,
            "sources": [],
            "reformulated_query": reformulated_query
        }

    # Combine retrieved chunks
    context = "\n\n".join(documents)

    is_custom_doc = bool(temp_session_id and has_temp_session(temp_session_id))

    # Generate answer using original user query and retrieved context
    answer = generate_answer(
        query,
        context,
        is_custom_doc=is_custom_doc
    )

    return {
        "answer": answer,
        "sources": sources,
        "reformulated_query": reformulated_query
    }


def ask_question_stream(query, temp_session_id: str = None):
    # Step 1: Reformulate query for semantic search (with safe fallback)
    reformulated_query = reformulate_query(query)

    # Step 2: Retrieve relevant chunks (Temporary session vs Permanent DB)
    if temp_session_id and has_temp_session(temp_session_id):
        results = retrieve_temp_documents(
            session_id=temp_session_id,
            query=query,
            n_results=5,
            reformulated_query=reformulated_query
        )
        fallback_msg = NO_TEMP_FOUND_MSG
    else:
        results = retrieve_documents(
            query,
            n_results=5,
            reformulated_query=reformulated_query
        )
        fallback_msg = NO_POLICY_FOUND_MSG

    documents, sources = _filter_results_by_threshold(results, DISTANCE_THRESHOLD)

    # Log retrieval metrics for observability
    top_dist = results["distances"][0][0] if results and results.get("distances") and results["distances"][0] else None
    logger.info(f"[RAG Pipeline] TempSession: {temp_session_id} | Original: '{query}' | Reformulated: '{reformulated_query}' | Top Distance: {top_dist}")

    # No relevant context found within threshold
    if not documents:
        def fallback_stream():
            yield fallback_msg
        return {
            "stream": fallback_stream(),
            "sources": [],
            "reformulated_query": reformulated_query
        }

    # Combine retrieved chunks
    context = "\n\n".join(documents)

    is_custom_doc = bool(temp_session_id and has_temp_session(temp_session_id))

    # Stream answer generator using original user query and retrieved context
    stream = generate_answer_stream(
        query,
        context,
        is_custom_doc=is_custom_doc
    )

    return {
        "stream": stream,
        "sources": sources,
        "reformulated_query": reformulated_query
    }