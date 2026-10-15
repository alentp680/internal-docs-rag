from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


model = CrossEncoder(MODEL_NAME)


def rerank(
    query: str,
    documents: list[str],
    top_k: int = 3,
) -> list[tuple[int, float]]:

    pairs = [
        [query, document]
        for document in documents
    ]

    scores = model.predict(pairs)

    ranked_results = sorted(
        enumerate(scores),
        key=lambda item: item[1],
        reverse=True,
    )

    return ranked_results[:top_k]