def reciprocal_rank_fusion(
    rankings: list[list[int]],
    k: int = 60,
) -> list[tuple[int, float]]:

    scores = {}

    for ranking in rankings:

        for rank, document_id in enumerate(
            ranking,
            start=1,
        ):

            if document_id not in scores:
                scores[document_id] = 0.0

            scores[document_id] += 1 / (
                k + rank
            )

    return sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

def hybrid_search(
    dense_ranking: list[int],
    bm25_ranking: list[int],
    top_k: int = 3,
) -> list[tuple[int, float]]:
    """Combine dense and BM25 rankings using RRF."""

    results = reciprocal_rank_fusion(
        [
            dense_ranking,
            bm25_ranking,
        ]
    )

    return results[:top_k]