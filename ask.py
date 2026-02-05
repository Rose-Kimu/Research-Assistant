# ask.py - RAG System with Verification
import chromadb
from groq import Groq
import config
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- SETUP ---
# Ensure the database directory exists
Path(config.CHROMA_PATH).mkdir(parents=True, exist_ok=True)

# Initialize Client
chroma_client = chromadb.PersistentClient(path=config.CHROMA_PATH)

# Safe collection loading
try:
    collection = chroma_client.get_collection(config.COLLECTION_NAME)
except chromadb.errors.NotFoundError:
    collection = chroma_client.create_collection(config.COLLECTION_NAME)

# --- HELPER FUNCTIONS ---

def retrieve_context(user_query):
    """
    Fetches relevant text chunks from ChromaDB and joins them into a string.
    Enhanced with context quality assessment.
    """
    results = collection.query(
        query_texts=[user_query],
        n_results=5 
    )

    if not results['documents'] or not results['documents'][0]:
        return None
    
    # Get similarity scores if available
    distances = results.get('distances', [[]])[0] if results.get('distances') else []
    
    # Flatten list of lists and join
    context_text = "\n\n".join(results['documents'][0])
    
    # Assess context quality based on similarity scores
    if distances:
        avg_distance = sum(distances) / len(distances)
        if avg_distance > 0.8:  # High distance = low similarity
            print(f"   [Warning] Retrieved context may be of low relevance (avg distance: {avg_distance:.2f})")
    
    return context_text

def generate_draft(user_query, context_text):
    """
    Generates the initial answer using the context.
    """
    system_prompt = f"""
    You are a helpful assistant. You answer questions about research. 
    But you only answer based on knowledge I'm providing you. You don't use your internal 
    knowledge and you don't make things up.
    If you don't know the answer, just say: I don't know
    --------------------
    The data:
    {context_text}
    """

    response = groq_client.chat.completions.create(
        model=config.MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
    )
    
    return response.choices[0].message.content

def verify_groundedness(draft_response, context):
    """
    The Judge: Checks if the draft is actually supported by the context.
    Implements Evidence Support Score as fraction of key factual claims supported.
    """
    print(f"   [Guardrail] Verifying draft against {len(context.split())} words of context...")
    
    judge_prompt = f"""
    You are a Fact-Checking Judge implementing an Evidence Support Score system.
    
    CONTEXT (Source Material):
    {context}
    
    DRAFT RESPONSE:
    {draft_response}
    
    Your Task:
    1. IDENTIFY all key factual claims in the draft response (ignore opinions, greetings, or general statements)
    2. For EACH claim, check if it is EXPLICITLY supported by the context
    3. CALCULATE the Evidence Support Score as: (Number of supported claims) / (Total key factual claims)
    4. PROVIDE detailed reasoning showing which claims are supported vs unsupported
    
    Scoring Guidelines:
    - 1.0 = All key factual claims are explicitly supported by context
    - 0.7-0.9 = Most claims supported, minor unsupported details
    - 0.3-0.6 = Some claims supported, significant unsupported content
    - 0.0-0.2 = Major hallucinations, most claims unsupported
    
    Return ONLY a JSON object with this exact format:
    {{
        "score": <float between 0.0 and 1.0>,
        "total_claims": <integer>,
        "supported_claims": <integer>,
        "reasoning": "<detailed explanation of which claims are supported/unsupported>",
        "confidence": "<high/medium/low based on context quality>"
    }}
    """

    response = groq_client.chat.completions.create(
        model=config.MODEL_NAME,
        messages=[{'role': 'user', 'content': judge_prompt}],
        response_format={"type": "json_object"}
    )
    
    try:
        result = json.loads(response.choices[0].message.content)
        # Ensure score is within bounds
        result['score'] = max(0.0, min(1.0, result.get('score', 0.0)))
        return result
    except json.JSONDecodeError:
        return {
            "score": 0.0, 
            "total_claims": 0,
            "supported_claims": 0,
            "reasoning": "Error: Judge failed to output valid JSON.",
            "confidence": "low"
        }

def self_correct(user_query, context, draft, reasoning):
    """
    Rewrites the answer if the Evidence Support Score is low.
    Focuses on grounding all claims in the provided context.
    """
    print("   [Guardrail] Triggering Self-Correction...")
    
    correction_prompt = f"""
    Your previous answer was rejected due to insufficient grounding in the source material.
    
    ORIGINAL QUESTION:
    {user_query}
    
    PREVIOUS ANSWER (REJECTED):
    {draft}
    
    VERIFICATION FEEDBACK:
    {reasoning}
    
    AVAILABLE CONTEXT (Your ONLY source of information):
    {context}
    
    Instructions for Correction:
    1. Answer ONLY using information explicitly stated in the context above
    2. If the context doesn't contain enough information, say "Based on the available information, I can only tell you..." and provide what you can
    3. Do NOT add information from your training data
    4. If you cannot answer the question with the given context, clearly state this limitation
    5. Cite specific parts of the context when making claims
    
    Provide your corrected response:
    """
    
    response = groq_client.chat.completions.create(
        model=config.MODEL_NAME,
        messages=[{"role": "user", "content": correction_prompt}]
    )
    return response.choices[0].message.content

# --- MAIN LOGIC LOOP ---

def answer_question(user_query):
    print(f"\nThinking about: {user_query}...")

    # 1. Retrieve
    context_text = retrieve_context(user_query)
    
    if not context_text:
        return "I couldn't find any relevant information in the database."

    # 2. Draft
    draft_response = generate_draft(user_query, context_text)
    
    # 3. Verify (Self-Evaluation Node)
    verification_result = verify_groundedness(draft_response, context_text)
    score = verification_result.get('score', 0.0)
    total_claims = verification_result.get('total_claims', 0)
    supported_claims = verification_result.get('supported_claims', 0)
    reasoning = verification_result.get('reasoning', 'No reasoning provided')
    confidence = verification_result.get('confidence', 'unknown')
    
    print(f"   [Evidence Support Score: {score:.2f}] ({supported_claims}/{total_claims} claims supported)")
    print(f"   [Confidence: {confidence}] {reasoning}")

    # 4. Decide & Correct based on Evidence Support Score
    if score < 0.7:  # Threshold for acceptable groundedness
        print("   [Guardrail] Score below threshold - triggering self-correction...")
        final_response = self_correct(user_query, context_text, draft_response, reasoning)
        
        # Optional: Re-verify the corrected response
        re_verification = verify_groundedness(final_response, context_text)
        re_score = re_verification.get('score', 0.0)
        print(f"   [Re-verification Score: {re_score:.2f}] After correction")
        
        return final_response
    elif score < 0.9:
        # Medium confidence - add disclaimer
        return f"{draft_response}\n\n[Note: Response confidence is {confidence} based on available context]"
    
    return draft_response

if __name__ == "__main__":
    while True:
        q = input("\nEnter a question (or 'q' to quit): ")
        if q.lower() == 'q':
            break
        
        final_answer = answer_question(q)
        print(f"\n FINAL ANSWER:\n{final_answer}")