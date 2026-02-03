# ask.py
import chromadb
import ollama
import config

import chromadb
import config
from pathlib import Path

# Make sure the path exists
Path(config.CHROMA_PATH).mkdir(parents=True, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=config.CHROMA_PATH)

# Safe collection creation
try:
    collection = chroma_client.get_collection(config.COLLECTION_NAME)
except chromadb.errors.NotFoundError:
    collection = chroma_client.create_collection(config.COLLECTION_NAME)


# Initialize client once at module level or inside function
# chroma_client = chromadb.PersistentClient(path=config.CHROMA_PATH)
# collection = chroma_client.get_or_create_collection(name=config.COLLECTION_NAME)

def answer_question(user_query):
    """
    Retrieves context from ChromaDB and queries Ollama.
    Returns the response string.
    """
    print(f"\nThinking about: {user_query}...")

    # 1. Retrieve Context
    results = collection.query(
        query_texts=[user_query],
        n_results=5 # Increased to 3 for better context
    )

    if not results['documents'][0]:
        return "I couldn't find any relevant information in the database."

    # Flatten the list of lists returned by Chroma
    context_text = "\n\n".join(results['documents'][0])

    # 2. Construct Prompt
    system_prompt = f"""
    You are a helpful assistant. You answer questions about research. 
    But you only answer based on knowledge I'm providing you. You don't use your internal 
    knowledge and you don't make things up.
    If you don't know the answer, just say: I don't know
    --------------------
    The data:
    {context_text}
    """

    # 3. Generate Response
    response = ollama.chat(
        model=config.MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
    )

    return response["message"]["content"]

if __name__ == "__main__":
    # Test block
    q = input("Enter a test question: ")
    print(answer_question(q))