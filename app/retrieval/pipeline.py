from app.retrieval.document_index import document_index
from qdrant_client import QdrantClient

from app.embeddings.model import create_embedding
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.hybrid import hybrid_search
from app.retrieval.reranker import rerank


client = QdrantClient(
    host="localhost",
    port=6333
)

collection_name = "internal_docs"


def retrieve(
    query: str,
    top_k: int = 3,
):
    chunks = document_index.get_all_chunks()

    
    """
    Retrieve the most relevant chunks for a query
    from all indexed documents.
    """

    # 1. Create query embedding
    query_embedding = create_embedding(query)

    # 2. Dense search in Qdrant
    dense_results = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=top_k,
    )

    dense_ranking = []

    for result in dense_results.points:

        chunk_id = result.payload["metadata"]["chunk_id"]

        for index, chunk in enumerate(chunks):

            if chunk["metadata"]["chunk_id"] == chunk_id:

                dense_ranking.append(index)
                break

    # 3. BM25 search across ALL chunks

    if document_index.bm25 is None:

        document_index.bm25 = BM25Retriever(
            chunks
        )

    bm25_ranking = document_index.bm25.search(
    query,
    top_k=top_k,
    )

    # 4. Combine dense + BM25 rankings
    hybrid_results = hybrid_search(
        dense_ranking,
        bm25_ranking,
        top_k=top_k,
    )

    hybrid_chunk_indices = [
        index
        for index, score in hybrid_results
    ]

    # 5. Get candidate documents
    candidate_documents = [
        chunks[index]["text"]
        for index in hybrid_chunk_indices
    ]

    # 6. Rerank candidates
    reranked_results = rerank(
        query,
        candidate_documents,
        top_k=top_k,
    )

    # 7. Build final results
    results = []

    for index, score in reranked_results:

        original_index = hybrid_chunk_indices[index]

        results.append(
            {
                "text": chunks[original_index]["text"],
                "metadata": chunks[original_index]["metadata"],
                "score": float(score),
            }
        )

    return results