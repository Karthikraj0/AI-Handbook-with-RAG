import ollama


EMBEDDING_MODEL = "nomic-embed-text"


def create_embedding(text):

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=text,
        keep_alive="1h"
    )

    return response["embeddings"][0]


def create_embeddings(texts):

    embeddings = []

    for text in texts:

        embedding = create_embedding(text)

        embeddings.append(embedding)

    return embeddings