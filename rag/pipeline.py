from rag.retrieve import retrieve_documents
from rag.generator import generate_answer, generate_answer_stream


DISTANCE_THRESHOLD = 0.47
NO_POLICY_FOUND_MSG = "I couldn't find that information in the company policies."


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


def ask_question(query):

    # Retrieve relevant policy chunks
    results = retrieve_documents(
        query,
        n_results=3
    )

    documents, sources = _filter_results_by_threshold(results, DISTANCE_THRESHOLD)

    # No relevant policy found within threshold
    if not documents:
        return {
            "answer": NO_POLICY_FOUND_MSG,
            "sources": []
        }

    # Combine retrieved chunks
    context = "\n\n".join(documents)

    # Generate answer using GPT-OSS
    answer = generate_answer(
        query,
        context
    )

    return {
        "answer": answer,
        "sources": sources
    }


def ask_question_stream(query):

    # Retrieve relevant policy chunks
    results = retrieve_documents(
        query,
        n_results=3
    )

    documents, sources = _filter_results_by_threshold(results, DISTANCE_THRESHOLD)

    # No relevant policy found within threshold
    if not documents:
        def fallback_stream():
            yield NO_POLICY_FOUND_MSG
        return {
            "stream": fallback_stream(),
            "sources": []
        }

    # Combine retrieved chunks
    context = "\n\n".join(documents)

    # Stream answer generator using GPT-OSS
    stream = generate_answer_stream(
        query,
        context
    )

    return {
        "stream": stream,
        "sources": sources
    }