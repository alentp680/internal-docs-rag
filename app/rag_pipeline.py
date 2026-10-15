from app.retrieval.pipeline import retrieve
from app.llm.ollama_client import generate_answer


def ask(
    query: str,
    top_k: int = 3,
):
    """
    Answer a question using all indexed document chunks.
    """

    # 1. Retrieve relevant chunks
    results = retrieve(
        query=query,
        top_k=top_k,
    )

    # 2. Build context
    context = "\n\n".join(
        result["text"]
        for result in results
    )

    # 3. Generate answer
    answer = generate_answer(
        query=query,
        context=context,
    )

    # 4. Collect sources
    sources = []

    for result in results:

        metadata = result["metadata"]

        source = {
            "source": metadata["source"],
            "file_type": metadata["file_type"],
            "chunk_id": metadata["chunk_id"],
        }

        if "page" in metadata:
            source["page"] = metadata["page"]

        sources.append(source)

    return {
        "answer": answer,
        "sources": sources,
    }