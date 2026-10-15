from qdrant_client import QdrantClient

from qdrant_client.models import Distance, VectorParams


client = QdrantClient(
    host="localhost",
    port=6333
)


collection_name = "internal_docs"


if not client.collection_exists(collection_name):

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )

    print(
        f"Collection '{collection_name}' created successfully!"
    )

else:

    print(
        f"Collection '{collection_name}' already exists."
    )