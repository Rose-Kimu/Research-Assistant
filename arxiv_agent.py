# arxiv_agent.py - ArXiv API Integration with Verification
import feedparser
from groq import Groq
from urllib.parse import urlencode
import config
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

ARXIV_API = "http://export.arxiv.org/api/query"

def verify_arxiv_groundedness(user_query, papers, generated_response):
    """
    Verifies if the generated response about ArXiv papers is grounded in the actual paper data.
    """
    # Combine all paper abstracts and titles as the "context"
    arxiv_context = ""
    for i, paper in enumerate(papers, 1):
        arxiv_context += f"Paper {i}: {paper['title']}\nAbstract: {paper['abstract']}\n\n"
    
    judge_prompt = f"""
    You are a Fact-Checking Judge for ArXiv paper summaries.
    
    USER QUERY:
    {user_query}
    
    ARXIV PAPER DATA (Source Material):
    {arxiv_context}
    
    GENERATED RESPONSE:
    {generated_response}
    
    Your Task:
    1. IDENTIFY all factual claims in the generated response
    2. Check if each claim is EXPLICITLY supported by the ArXiv paper data
    3. CALCULATE the Evidence Support Score as: (Supported claims) / (Total claims)
    
    Scoring Guidelines:
    - 1.0 = All claims are directly from the paper titles/abstracts
    - 0.7-0.9 = Most claims supported, minor interpretation
    - 0.3-0.6 = Some claims supported, some speculation
    - 0.0-0.2 = Major hallucinations, claims not in papers
    
    Return ONLY a JSON object:
    {{
        "score": <float between 0.0 and 1.0>,
        "total_claims": <integer>,
        "supported_claims": <integer>,
        "reasoning": "<explanation of which claims are/aren't supported>",
        "confidence": "<high/medium/low based on paper data quality>"
    }}
    """
    
    response = groq_client.chat.completions.create(
        model=config.MODEL_NAME,
        messages=[{'role': 'user', 'content': judge_prompt}],
        response_format={"type": "json_object"}
    )
    
    try:
        import json
        result = json.loads(response.choices[0].message.content)
        result['score'] = max(0.0, min(1.0, result.get('score', 0.0)))
        return result
    except json.JSONDecodeError:
        return {
            "score": 0.0,
            "total_claims": 0,
            "supported_claims": 0,
            "reasoning": "Error: Judge failed to output valid JSON for ArXiv verification.",
            "confidence": "low"
        }

def search_arxiv(topic, max_results=5):
    """
    Searches arXiv for a specific topic and returns a list of paper metadata.
    """
    params = {
        "search_query": f"all:{topic}",
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": max_results
    }
    url = f"{ARXIV_API}?{urlencode(params)}"
    feed = feedparser.parse(url)
    
    papers = []
    for entry in feed.entries:
        papers.append({
            "title": entry.title.replace("\n", " "),
            "abstract": entry.summary.replace("\n", " "),
            "link": entry.id
        })
    return papers

def generate_arxiv_summary(user_query, papers):
    """
    Generate a comprehensive summary of ArXiv papers with groundedness verification.
    """
    if not papers:
        return "I searched arXiv but found no papers on this topic.", None
    
    # Create context from papers
    papers_context = ""
    for i, paper in enumerate(papers, 1):
        papers_context += f"Paper {i}: {paper['title']}\nAbstract: {paper['abstract']}\nLink: {paper['link']}\n\n"
    
    # Generate summary
    summary_prompt = f"""
    You are a research assistant. Based on the ArXiv papers below, provide a comprehensive summary that addresses the user's query.
    
    USER QUERY: {user_query}
    
    ARXIV PAPERS:
    {papers_context}
    
    Instructions:
    1. Summarize the key findings relevant to the user's query
    2. Mention specific paper titles when referencing findings
    3. Only include information explicitly stated in the abstracts
    4. If papers don't directly address the query, state this clearly
    
    Provide a clear, informative summary:
    """
    
    response = groq_client.chat.completions.create(
        model=config.MODEL_NAME,
        messages=[{"role": "user", "content": summary_prompt}]
    )
    
    generated_summary = response.choices[0].message.content
    
    # Verify groundedness
    verification_result = verify_arxiv_groundedness(user_query, papers, generated_summary)
    
    return generated_summary, verification_result
    """
    Verifies if the generated response about ArXiv papers is grounded in the actual paper data.
    """
    # Combine all paper abstracts and titles as the "context"
    arxiv_context = ""
    for i, paper in enumerate(papers, 1):
        arxiv_context += f"Paper {i}: {paper['title']}\nAbstract: {paper['abstract']}\n\n"
    
    judge_prompt = f"""
    You are a Fact-Checking Judge for ArXiv paper summaries.
    
    USER QUERY:
    {user_query}
    
    ARXIV PAPER DATA (Source Material):
    {arxiv_context}
    
    GENERATED RESPONSE:
    {generated_response}
    
    Your Task:
    1. IDENTIFY all factual claims in the generated response
    2. Check if each claim is EXPLICITLY supported by the ArXiv paper data
    3. CALCULATE the Evidence Support Score as: (Supported claims) / (Total claims)
    
    Scoring Guidelines:
    - 1.0 = All claims are directly from the paper titles/abstracts
    - 0.7-0.9 = Most claims supported, minor interpretation
    - 0.3-0.6 = Some claims supported, some speculation
    - 0.0-0.2 = Major hallucinations, claims not in papers
    
    Return ONLY a JSON object:
    {{
        "score": <float between 0.0 and 1.0>,
        "total_claims": <integer>,
        "supported_claims": <integer>,
        "reasoning": "<explanation of which claims are/aren't supported>",
        "confidence": "<high/medium/low based on paper data quality>"
    }}
    """
    
    response = groq_client.chat.completions.create(
        model=config.MODEL_NAME,
        messages=[{'role': 'user', 'content': judge_prompt}],
        response_format={"type": "json_object"}
    )
    
    try:
        import json
        result = json.loads(response.choices[0].message.content)
        result['score'] = max(0.0, min(1.0, result.get('score', 0.0)))
        return result
    except json.JSONDecodeError:
        return {
            "score": 0.0,
            "total_claims": 0,
            "supported_claims": 0,
            "reasoning": "Error: Judge failed to output valid JSON for ArXiv verification.",
            "confidence": "low"
        }
    """
    Uses Groq to summarize a paper and score its relevance.
    """
    system_prompt = f"""
    You are a research assistant.
    Task: Summarize the paper in 3 bullet points and score relevance (0-3) to topic: "{topic}".
    Format:
    SUMMARY:
    - Point 1
    - Point 2
    SCORE: <number>
    """
    
    user_prompt = f"TITLE: {paper['title']}\nABSTRACT: {paper['abstract']}"
    
    response = groq_client.chat.completions.create(
        model=config.MODEL_NAME,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    )
    
    content = response.choices[0].message.content
    
    # Robust Parsing
    try:
        # Split by the specific keyword "SCORE:"
        parts = content.split("SCORE:")
        summary = parts[0].replace("SUMMARY:", "").strip()
        score_text = parts[1].strip()
        # Handle cases where score might have punctuation like "3."
        score = int(''.join(filter(str.isdigit, score_text)))
    except:
        # Fallback if the LLM output is messy
        summary = content
        score = 0
        
    return summary, score

def summarize_and_score(paper, topic):
    """
    Uses Groq to summarize a paper and score its relevance.
    """
    system_prompt = f"""
    You are a research assistant.
    Task: Summarize the paper in 3 bullet points and score relevance (0-3) to topic: "{topic}".
    Format:
    SUMMARY:
    - Point 1
    - Point 2
    SCORE: <number>
    """
    
    user_prompt = f"TITLE: {paper['title']}\nABSTRACT: {paper['abstract']}"
    
    response = groq_client.chat.completions.create(
        model=config.MODEL_NAME,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    )
    
    content = response.choices[0].message.content
    
    # Robust Parsing
    try:
        # Split by the specific keyword "SCORE:"
        parts = content.split("SCORE:")
        summary = parts[0].replace("SUMMARY:", "").strip()
        score_text = parts[1].strip()
        # Handle cases where score might have punctuation like "3."
        score = int(''.join(filter(str.isdigit, score_text)))
    except:
        # Fallback if the LLM output is messy
        summary = content
        score = 0
        
    return summary, score