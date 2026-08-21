def chunk_documents(documents, chunk_size=150, overlap=30):

    chunks = []

    for document in documents:

        words = document["text"].split()

        start = 0

        while start < len(words):

            end = start + chunk_size

            chunk_text = " ".join(words[start:end])

            chunks.append({
                "text": chunk_text,
                "source": document["source"],
                "page": document["page"]
            })

            start += chunk_size - overlap

    return chunks