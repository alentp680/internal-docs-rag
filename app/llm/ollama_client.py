from ollama import chat


MODEL_NAME = "qwen2.5:3b"


def generate_answer(
    query: str,
    context: str,
) -> str:

    prompt = f"""
You are a helpful internal document assistant.

Answer the user's question using only the provided context.

If the answer cannot be found in the context, say:
"I could not find the answer in the provided documents."

Context:
{context}

Question:
{query}

Answer:
"""

    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response.message.content