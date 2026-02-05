# Technical Brief: Research-Focused RAG Agent

**Course:** Agentic AI Fundamentals and Applications  
**Assignment:** Building the Agent  
**Date:** February 5, 2025

## Project Overview

This project implements a RAG-enabled agent for academic research assistance. The system combines three core modules: a retrieval system using ChromaDB, an ArXiv API tool for external paper search, and a two-step verification system for groundedness scoring. The agent uses ReAct-style reasoning to decide between local database queries and external tool calls.

## Tooling Rationale

### External Tool: ArXiv API Integration

**Tool Description:**
We built an ArXiv API integration tool that searches for academic papers and provides summaries with relevance scoring. The tool consists of two main functions:
- `search_arxiv()`: Queries the ArXiv API and retrieves paper metadata
- `generate_arxiv_summary()`: Creates summaries with groundedness verification

**Why This Tool is Necessary:**
The ArXiv tool is necessary for our project's goals because:

1. **Knowledge Gap Coverage**: Our local database contains only 5 research papers. For a research assistant, users will frequently ask about topics not covered in the local documents. The ArXiv tool provides access to the broader research literature.

2. **Current Research Access**: Academic research moves quickly. Users need access to recent publications that wouldn't be in a static local database. ArXiv provides real-time access to the latest research papers.

3. **Domain Expansion**: Our local documents focus on specific areas (ML, CV, NLP, robotics, AI ethics). The ArXiv tool allows the agent to handle queries about any research domain, making it more useful as a research assistant.

4. **Verification Requirements**: For HW2, we needed to demonstrate tool-calling with verification. The ArXiv tool generates summaries that can be verified against the paper abstracts, showing how external tool outputs can be grounded in source material.

**Technical Implementation:**
The tool integrates with the ReAct reasoning loop through explicit decision criteria. The agent chooses ArXiv when users ask to "find papers," "search for research," or request information about topics not in the local database. The tool returns structured data (titles, abstracts, links) that can be verified for groundedness.

**Verification Integration:**
We added `verify_arxiv_groundedness()` to check if generated summaries are supported by the actual paper abstracts. This provides an ArXiv Evidence Support Score (0.0-1.0) similar to the local database verification, ensuring consistent quality control across both data sources.

## Failure Analysis

### Initial Failure: Incorrect Tool Selection

**Specific Instance:**
When we first tested the system, we asked: "What languages were tested in the multilingual sentiment analysis study?" The agent incorrectly chose the ArXiv search tool instead of the local database, even though this information was clearly available in our local documents.

**What Went Wrong:**
The agent's reasoning was: "I should search ArXiv for multilingual sentiment analysis studies to find which languages were tested." This happened because our initial tool instructions were too vague:

```python
# Original (problematic) instructions
TOOLS_INSTRUCTIONS = """
1. [local_db]: Use this to answer questions about local documents
2. [arxiv_search]: Use this to find research papers
"""
```

The agent saw "multilingual sentiment analysis study" and interpreted this as a request to find research papers, rather than a question about specific content from existing documents.

**Technical Adjustment Made:**
We enhanced the tool selection logic with priority-based routing and explicit decision criteria:

```python
# Fixed routing logic
# Priority 1: Explicit search requests go to ArXiv
if any(word in user_query_lower for word in ["find", "search", "recent", "new"]):
    action = "arxiv_search"
# Priority 2: Questions about specific content go to local DB  
elif any(word in user_query_lower for word in ["study", "research", "analysis", "multilingual", "computer vision", "bias", "ethics", "challenges", "findings", "summary", "about"]):
    action = "local_db"
# Priority 3: General "papers" requests go to ArXiv
elif "papers" in user_query_lower:
    action = "arxiv_search"
```

This priority-based system ensures that:
1. Explicit search intent ("find", "search", "recent", "new") always goes to ArXiv
2. Content-specific questions go to local database first
3. General paper requests default to ArXiv search

**Results After Fix:**
After implementing the priority-based routing system, the agent correctly identified that "What languages were tested in the multilingual sentiment analysis study?" was asking for specific data from existing documents, not requesting new paper searches. The success rate for correct tool selection improved  for local queries.

The new system handles edge cases better:
- "Find papers on computer vision challenges" → ArXiv (explicit search intent)
- "What are the challenges in computer vision?" → Local DB (content-specific question)
- "Give me a summary of the bias study" → Local DB (specific content request)

**Additional Verification Enhancement:**
This failure also highlighted the need for better verification of tool outputs. We added re-verification after self-correction and implemented the dual groundedness system to catch similar issues across both local and external data sources.

## System Performance

The final system demonstrates:
- **Tool Selection Accuracy**: for both local queries and ArXiv queries
- **Verification Coverage**: 100% of responses receive groundedness scores
- **Self-Correction**: Automatically improves responses with scores below 0.7
- **Dual Verification**: Both local database and ArXiv tool outputs are verified

The combination of improved tool selection logic and comprehensive verification ensures the agent provides reliable, grounded responses for academic research assistance.



**Contribution Statement:**
This implementation represents group work demonstrating understanding of RAG architecture, external tool integration, ReAct reasoning, and verification systems as required for HW2.