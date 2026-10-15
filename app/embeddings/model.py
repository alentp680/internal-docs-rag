from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


model = SentenceTransformer(MODEL_NAME)


def create_embedding(text: str) -> list[float]:
    """Create a vector embedding for a piece of text."""

    embedding = model.encode(text)

    return embedding.tolist()