# Research-Focused RAG Agent - HW2 Implementation

This project implements a  RAG-enabled agent for academic research assistance with groundedness verification.

## Architecture Overview

The system consists of three core modules with enhanced verification:

1. **Retrieval Module (Memory)** - ChromaDB vector database with research papers
2. **Tool-Calling Module** - ArXiv API integration for real-time paper discovery  
3. **Verification Module (Guardrails)** - Evidence scoring for local and external data

## Groundedness System

### Local Database Verification
- **Evidence Support Score:** Verifies claims against document chunks
- **Self-Correction:** Automatic rewriting for scores < 0.7
- **Re-verification:** Quality assurance after corrections

### ArXiv Data Verification
- **ArXiv Evidence Support Score:** Verifies claims against paper abstracts
- **Context Quality Assessment:** Evaluates paper data completeness
- **Confidence Scoring:** High/medium/low based on source quality

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Setup
1. Add your Groq API key to `.env`:
   ```
   GROQ_API_KEY=your_actual_groq_api_key_here
   ```

2. Populate the vector database:
   ```bash
   python fill_db.py
   ```

### Running the Agent

**Interactive Chat Mode:**
```bash
python design.py
```

**Q&A Mode (Retrieval + Verification):**
```bash
python ask.py
```

**Full Demo (HW2 Requirements):**
```bash
python hw2_automated_demo.py
```

## Project Modules

### 1. Retrieval Module 
- **Domain-Specific Ingestion:** Research papers in ChromaDB
- **Advanced Chunking:** RecursiveCharacterTextSplitter (300 chars, 100 overlap)
- **File:** `fill_db.py`, `ask.py`

### 2. Tool-Calling Module 
- **External Tool:** ArXiv API for paper search
- **ReAct Loop:** Intelligent tool selection in `design.py`
- **File:** `arxiv_agent.py`, `design.py`

### 3. Verification Module 
- **Local Evidence Scoring:** 0.0-1.0 scale for database responses
- **ArXiv Evidence Scoring:** 0.0-1.0 scale for external tool outputs
- **Dual Verification Coverage:** 100% of responses verified
- **File:** `ask.py`, `arxiv_agent.py`

### 4. Documentation 
- **Technical Brief:** `HW2_Technical_Brief.md`
- **Implementation Trace:** `HW2_Implementation_Trace.log`
- **Code Comments:** Throughout all modules

## Key Files

- `config.py` - Central configuration
- `ask.py` - Retrieval + Local verification system
- `arxiv_agent.py` - ArXiv API tool + ArXiv verification
- `design.py` - Main ReAct agent with tool selection
- `fill_db.py` - Database ingestion
- `hw2_automated_demo.py` - Complete system demonstration
- `HW2_Technical_Brief.md` - Technical documentation

## Testing Queries

Try these queries to see different verification systems in action:

**Local Database Verification:**
- "What are the key findings about bias in large language models?"
- "What accuracy improvements were mentioned for climate model interpretation?"
- "What languages were tested in the multilingual sentiment analysis study?"

**ArXiv Tool Verification:**  
- "Find recent papers about transformer attention mechanisms"
- "Search for new research on reinforcement learning in robotics"
- "What are the latest developments in quantum machine learning?"

**Expected Output Indicators:**
- Local queries: `[Evidence Support Score: X.XX]`
- ArXiv queries: `[ArXiv Evidence Support Score: X.XX]`
- Both show confidence levels and verification reasoning

## Technical Highlights

- **LLM:** Llama 3.3 70B hosted on Groq for fast inference
- **Vector DB:** ChromaDB with persistent storage
- **Chunking:** Context-preserving strategy for research content
- **Dual Verification:** Evidence scoring for both local and external data
- **Tool Selection:** ReAct-style reasoning with explicit decision criteria
- **Transparency:** Complete verification traces for all responses



## Verification System Features

### Local Database Verification
- Claim-by-claim analysis against document chunks
- Automatic self-correction for low scores
- Re-verification after corrections
- Context quality warnings

### ArXiv Data Verification
- Summary verification against paper abstracts
- Confidence assessment based on paper data quality
- Transparent source attribution
- Quality-based response disclaimers

## Decision Making Demo

The agent shows clear decision-making:

**Local Database Route:**
```
🧠 Agent is thinking...
👉 Decided to check Local Database for: 'bias reduction percentages'
[Evidence Support Score: 0.95] (3/3 claims supported)
```

**ArXiv Tool Route:**
```
🧠 Agent is thinking...
👉 Decided to search ArXiv for: 'recent transformer papers'
[ArXiv Evidence Support Score: 0.82] (4/5 claims supported)
```

## Implementation Details

The system initially had issues with tool selection - it choosing ArXiv for local document queries instead of using documents in locan storage. This was resolved by enhancing the reasoning prompt with explicit decision criteria and adding stricter verification for both data sources.

See `HW2_Technical_Brief.md` for detailed technical analysis and implementation decisions.

---
