
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from app.retrieval.document_index import document_index
from app.ingestion.loader import load_document
from app.ingestion.chunker import create_chunks
from app.embeddings.model import create_embedding


client = QdrantClient(
    host="localhost",
    port=6333
)

collection_name = "internal_docs"


def ingest_document(file_path: str) -> int:
    """Load, chunk, embed, and store a document in Qdrant."""

    documents = load_document(file_path)
    chunks = create_chunks(documents)

    points = []

    for chunk in chunks:
        embedding = create_embedding(chunk["text"])

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "text": chunk["text"],
                "metadata": chunk["metadata"],
            },
        )

        points.append(point)

    if points:
        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )

    document_index.add_document(file_path, chunks)

    return len(points)


def delete_document_from_qdrant(filename: str) -> bool:
    """Delete all Qdrant points belonging to a document."""

    result = client.delete(
        collection_name=collection_name,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="metadata.source",
                    match=MatchValue(value=filename),
                )
            ]
        ),
        wait=True,
    )

    return result.status == "completed"
