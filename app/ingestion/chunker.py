from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunks(
    documents: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[dict]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = []

    for document in documents:

        text = document["text"]
        metadata = document["metadata"]

        split_texts = splitter.split_text(text)

        for index, chunk_text in enumerate(split_texts):

            chunk_metadata = metadata.copy()

            chunk_metadata["chunk_id"] = f"{metadata['source']}_{index}"

            chunks.append(
                {
                    "text": chunk_text,
                    "metadata": chunk_metadata,
                }
            )

    return chunks