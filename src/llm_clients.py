import os
import ollama
from dotenv import load_dotenv
from google import genai
from google.genai import types  # Required for configuration
import config # Assuming this holds your MODEL_NAME and GEMINI_MODEL

load_dotenv()

# 1. Initialize Clients
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Gemini Client (replacing openai_client)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

def ollama_chat(system_prompt, user_prompt):
    """Chat using local Ollama models."""
    response = ollama.chat(
        model=config.MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response["message"]["content"]

def gemini_chat(system_prompt, user_prompt):
    client = genai.Client(api_key=GEMINI_API_KEY)
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=system_prompt+user_prompt
        )
        print(f"Gemini response: {response.text}")
        return response.text
    except Exception as e:
        return f"Gemini Error: {str(e)}"
