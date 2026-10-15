from qdrant_client import QdrantClient

class DocumentIndex:

    def __init__(self):
        self.documents = []
        self.chunks = []
        self.bm25 = None

    def add_document(
        self,
        file_path: str,
        chunks: list[dict],
    ):
        """
        Add a document and its chunks to the index.
        """

        self.documents.append(file_path)

        self.chunks.extend(chunks)

        self.bm25 = None

    def add_chunks(
        self,
            chunks: list[dict],
    ):
        """
        Add existing chunks to the index.
        """

        self.chunks.extend(chunks)

        self.bm25 = None

    
    def remove_document(self, filename: str):
        """Remove a document and its chunks from the in-memory index."""

        self.documents = [
            source for source in self.documents
            if source.replace("\\", "/").split("/")[-1] != filename
        ]

        self.chunks = [
            chunk for chunk in self.chunks
            if chunk["metadata"]["source"] != filename
        ]

        self.bm25 = None


    def get_all_chunks(self) -> list[dict]:
        """
        Return all indexed chunks.
        """

        return self.chunks

    def get_chunks_for_bm25(self) -> list[dict]:
        """
        Return all chunks for BM25 indexing.
        """

        return self.chunks

    def get_document_count(self) -> int:
        """
        Return the number of indexed documents.
        """

        return len(self.documents)

    def clear(self):
        """
        Remove all documents and chunks from the index.
        """

        self.documents.clear()
        self.chunks.clear()



def load_from_qdrant():
    """
    Load all indexed chunks from Qdrant
    into the shared document index.
    """

    client = QdrantClient(
        host="localhost",
        port=6333
    )

    collection_name = "internal_docs"

    document_index.clear()

    offset = None

    while True:

        points, next_offset = client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:

            payload = point.payload

            chunk = {
                "text": payload["text"],
                "metadata": payload["metadata"],
            }

            document_index.add_chunks(
                [chunk]
            )
            source = chunk["metadata"]["source"]

            if source not in document_index.documents:

                document_index.documents.append(
                    source
                )
        if next_offset is None:
            break

        offset = next_offset

    return len(
        document_index.get_all_chunks()
    )

document_index = DocumentIndex()

load_from_qdrant()