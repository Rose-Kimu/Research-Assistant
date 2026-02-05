# design.py - Main ReAct Agent
from groq import Groq
import ask          # Retrieval module
import arxiv_agent  # Tool-calling module
import config
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Define the tools available to the agent
TOOLS_INSTRUCTIONS = """
You are a smart research assistant with access to the following tools:

1. [local_db]: Use this to answer questions about SPECIFIC content from documents we have stored locally. 
   Use when the question asks about specific data, results, or details from existing research papers.
   - Input: A specific question string.
   
2. [arxiv_search]: Use this to find NEW research papers from ArXiv that are not in our local database.
   Use when the question asks to "find papers", "search for research", or requests new/recent publications.
   - Input: A search topic string (do NOT add years or date ranges).

3. [none]: Use this for greetings, general chat, or if you can answer without tools.

DECISION RULES:
- If question mentions "study", "paper", "research" AND asks about specific results/data → use [local_db]
- If question asks to "find", "search", "recent papers" → use [arxiv_search]
- Questions about "AI Ethics Bias study", "multilingual sentiment analysis", "computer vision research" → use [local_db]

FORMAT INSTRUCTIONS:
Response must be in this exact format:
THOUGHT: <Reasoning about which tool to use>
ACTION: <Tool Name>
INPUT: <Input for the tool>
"""

def route_and_execute(user_query):
    """
    The ReAct Loop:
    1. Thought (Decide)
    2. Action (Call Tool)
    3. Observation (Get Result)
    4. Final Answer (Respond to User)
    """
    
    # --- STEP 1: REASONING (THOUGHT) ---
    print("\n🧠 Agent is thinking...")
    
    # Use rule-based routing to avoid API timeout issues
    user_query_lower = user_query.lower()
    
    # Determine action based on keywords
    if any(word in user_query_lower for word in ["find", "search", "recent", "new", "papers"]) and not any(word in user_query_lower for word in ["challenges", "study", "analysis"]):
        action = "arxiv_search"
        tool_input = user_query
    elif any(word in user_query_lower for word in ["study", "research", "analysis", "multilingual", "computer vision", "bias", "ethics", "challenges", "findings"]):
        action = "local_db"
        tool_input = user_query
    else:
        action = "none"
        tool_input = user_query

    # --- STEP 2 & 3: ACTION & OBSERVATION ---
    
    # CASE A: Use Local RAG (Retrieval Module)
    if "local_db" in action:
        print(f"👉 Decided to check Local Database for: '{tool_input}'")
        try:
            observation = ask.answer_question(tool_input)
            return f"Based on your local documents:\n{observation}"
        except Exception as e:
            return f"Error accessing local database: {str(e)}"

    # CASE B: Use ArXiv API (Tool-Calling Module)
    elif "arxiv_search" in action:
        print(f"👉 Decided to search ArXiv for: '{tool_input}'")
        try:
            papers = arxiv_agent.search_arxiv(tool_input, max_results=3)
            
            if not papers:
                return "I searched arXiv but found no papers on this topic."
            
            # Generate summary with groundedness verification
            summary_text, verification_result = arxiv_agent.generate_arxiv_summary(tool_input, papers)
            
            if verification_result:
                score = verification_result.get('score', 0.0)
                total_claims = verification_result.get('total_claims', 0)
                supported_claims = verification_result.get('supported_claims', 0)
                reasoning = verification_result.get('reasoning', 'No reasoning provided')
                confidence = verification_result.get('confidence', 'unknown')
                
                print(f"   [ArXiv Evidence Support Score: {score:.2f}] ({supported_claims}/{total_claims} claims supported)")
                print(f"   [Confidence: {confidence}] {reasoning}")
                
                # Add verification info to response
                if score < 0.7:
                    summary_text += f"\n\n[Note: This summary has a low groundedness score ({score:.2f}). Please verify claims independently.]"
                elif score < 0.9:
                    summary_text += f"\n\n[Note: Summary confidence is {confidence} based on available paper abstracts.]"
            
            return summary_text
            
        except Exception as e:
            return f"I encountered an error searching arXiv: {str(e)}. Let me try to help with my local knowledge instead."

    # CASE C: No Tool Needed - Use Groq for general responses
    else:
        print("👉 Decided to provide a general response")
        try:
            response = groq_client.chat.completions.create(
                model=config.MODEL_NAME,
                messages=[{"role": "user", "content": f"Answer this question briefly: {user_query}"}],
                timeout=10,
                max_tokens=200
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"I can help with research questions. Try asking about specific studies, computer vision challenges, or search for new papers on arXiv."

def start_chat():
    print("==========================================")
    print("   🤖 AGENTIC AI ASSISTANT (ReAct Mode)   ")
    print("==========================================")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("\nUser: ").strip()
        if user_input.lower() == "exit":
            sys.exit()
            
        response = route_and_execute(user_input)
        print(f"\nAgent: {response}")
        print("-" * 40)

if __name__ == "__main__":
    start_chat()