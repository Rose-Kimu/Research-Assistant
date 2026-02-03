# arxiv_agent.py
import feedparser
import ollama
from urllib.parse import urlencode
import config  # Import central config

ARXIV_API = "http://export.arxiv.org/api/query"

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

def summarize_and_score(paper, topic):
    """
    Uses Ollama to summarize a paper and score its relevance.
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
    
    response = ollama.chat(
        model=config.MODEL_NAME,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    )
    
    content = response["message"]["content"]
    
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