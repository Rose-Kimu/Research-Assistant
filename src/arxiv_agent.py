import feedparser
from urllib.parse import urlencode
import config
from src.llm_clients import ollama_chat, gemini_chat


ARXIV_API = "http://export.arxiv.org/api/query"

# 🔁 SWITCH HERE
LLM_PROVIDER = "llama"   # "ollalma" or "openai"


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


def _call_llm(system_prompt, user_prompt):
    """
    Routes the prompt to the selected LLM provider.
    """
    if LLM_PROVIDER == "gemini":
        return gemini_chat(system_prompt, user_prompt)
    else:
        return ollama_chat(system_prompt, user_prompt)


def summarize_and_score(paper, topic):
    """
    Summarizes a paper and scores relevance using the selected LLM.
    """

    system_prompt = f"""
    You are a research assistant.
    Task: Summarize the paper in 3 bullet points and score relevance (0–3)
    to the topic: "{topic}"

    Format exactly like this:
    SUMMARY:
    - Point 1
    - Point 2
    - Point 3
    SCORE: <number>
    """

    user_prompt = f"""
    TITLE: {paper['title']}
    ABSTRACT: {paper['abstract']}
    """

    content = _call_llm(system_prompt, user_prompt)

    # 🔐 Robust parsing (LLM-safe)
    try:
        parts = content.split("SCORE:")
        summary = parts[0].replace("SUMMARY:", "").strip()
        score_text = parts[1].strip()
        score = int("".join(filter(str.isdigit, score_text)))
    except Exception:
        summary = content
        score = 0

    return summary, score
