from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self, chunks: list[dict]):

        self.chunks = chunks

        documents = [
            chunk["text"]
            for chunk in chunks
        ]

        tokenized_documents = [
            document.lower().split()
            for document in documents
        ]

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[int]:

        tokenized_query = query.lower().split()

        scores = self.bm25.get_scores(
            tokenized_query
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        return ranked_indices[:top_k]

    def get_chunk(
        self,
        index: int,
    ) -> dict:

        return self.chunks[index]