import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors

load_dotenv()
# IMPORTANT: Generate a NEW key in AI Studio since the last one was leaked!
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

def call_gemini():
    try:
        # Switching to the 2026 standard model: gemini-2.5-flash
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents="What are the key advancements in AI technology in 2024?"
        )
        print(f"Gemini response: {response.text}")
        
    except errors.ClientError as e:
        if e.status_code == 429:
            print("Quota exceeded! The Free Tier has a limit of 5 requests per minute.")
            print("Waiting 60 seconds before retrying...")
            time.sleep(60)
            call_gemini()
        else:
            print(f"An API error occurred: {e}")

if __name__ == "__main__":
    call_gemini()