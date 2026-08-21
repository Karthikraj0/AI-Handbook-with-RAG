from rag.retrieve import retrieve_documents
from rag.generator import generate_answer, generate_answer_stream


def ask_question(query):

    # Retrieve relevant policy chunks
    results = retrieve_documents(
        query,
        n_results=3
    )

    # No relevant policy found
    if results is None or not results.get("documents") or not results["documents"][0]:
        return {
            "answer": "I couldn't find that information in the company policies.",
            "sources": []
        }

    # Get retrieved text
    documents = results["documents"][0]

    # Combine retrieved chunks
    context = "\n\n".join(documents)

    # Generate answer using GPT-OSS
    answer = generate_answer(
        query,
        context
    )

    # Get source information
    sources = results["metadatas"][0] if results.get("metadatas") else []

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

    # No relevant policy found
    if results is None or not results.get("documents") or not results["documents"][0]:
        def fallback_stream():
            yield "I couldn't find that information in the company policies."
        return {
            "stream": fallback_stream(),
            "sources": []
        }

    # Get retrieved text
    documents = results["documents"][0]

    # Combine retrieved chunks
    context = "\n\n".join(documents)

    # Stream answer generator using GPT-OSS
    stream = generate_answer_stream(
        query,
        context
    )

    # Get source information
    sources = results["metadatas"][0] if results.get("metadatas") else []

    return {
        "stream": stream,
        "sources": sources
    }