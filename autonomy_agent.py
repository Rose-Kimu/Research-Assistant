"""
HW2: Agentic AI Research Assistant
04-801-W3 Agentic AI: Fundamentals and Applications

This autonomous agent integrates with  existing modules:
- ask.py (local database retrieval)
- arxiv_agent.py (ArXiv API tool)
- fill_db.py (database management)
- llm_clients.py (LLM providers)

Implements:
1. Retrieval Module - Uses your existing ChromaDB setup with enhanced chunking
2. Tool-Calling Module - ReAct loop with your arxiv_agent and ask modules
3. Verification Module - Groundedness scoring and hallucination detection
"""

import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import re
from pathlib import Path

# Import your existing modules
import src.ask as ask
import src.arxiv_agent as arxiv_agent
import src.fill_db as fill_db
import config
from src.llm_clients import gemini_chat, ollama_chat
import chromadb

# Configuration
LLM_PROVIDER = "llama3"  # or "ollama"

def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """Routes the prompt to the selected LLM provider."""
    if LLM_PROVIDER == "gemini":
        return gemini_chat(system_prompt, user_prompt)
    else:
        return ollama_chat(system_prompt, user_prompt)


# ============================================================================
# EXECUTION TRACE LOGGER
# ============================================================================

class ExecutionTrace:
    """Logs agent execution for transparency and debugging (Required for HW2)."""
    
    def __init__(self):
        self.trace = []
        self.start_time = datetime.now()
    
    def log_step(self, step_type: str, description: str, details: dict = None):
        """Log a single execution step."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "step_type": step_type,
            "description": description,
            "details": details or {}
        }
        self.trace.append(entry)
        
        # Print to console for real-time monitoring
        print(f"\n[{step_type.upper()}] {description}")
        if details:
            for key, value in details.items():
                print(f"  {key}: {value}")
    
    def save_trace(self, filename: str = "execution_trace.json"):
        """Save execution trace to file."""
        trace_data = {
            "session_start": self.start_time.isoformat(),
            "session_end": datetime.now().isoformat(),
            "total_steps": len(self.trace),
            "trace": self.trace
        }
        
        with open(filename, 'w') as f:
            json.dump(trace_data, f, indent=2)
        
        print(f"\n✅ Execution trace saved to {filename}")
        return filename


# ============================================================================
# MODULE 1: ENHANCED RETRIEVAL MODULE
# ============================================================================

class EnhancedRetriever:
    """
    Enhanced retrieval that wraps your existing ask.py functionality
    with semantic chunking and metadata tracking.
    """
    
    def __init__(self, trace: ExecutionTrace):
        self.trace = trace
        
        # Use your existing ChromaDB setup
        Path(config.CHROMA_PATH).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        
        try:
            self.collection = self.client.get_collection(config.COLLECTION_NAME)
            self.trace.log_step("initialization", "Retrieval module initialized", 
                              {"collection": config.COLLECTION_NAME, 
                               "document_count": self.collection.count()})
        except:
            self.collection = self.client.create_collection(config.COLLECTION_NAME)
            self.trace.log_step("initialization", "Created new collection", 
                              {"collection": config.COLLECTION_NAME})
    
    def semantic_chunk(self, text: str, chunk_size: int = 500, 
                      overlap: int = 100) -> List[Dict[str, str]]:
        """
        ADVANCED CHUNKING STRATEGY
        
        Uses sentence-boundary splitting with overlap to preserve semantic coherence.
        This is superior to fixed-size chunking for research papers.
        
        Justification:
        - Preserves sentence integrity (no mid-sentence cuts)
        - Maintains context through overlap
        - Optimal for research papers with complex ideas
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "length": len(chunk_text)
                })
                
                # Create overlap for context continuity
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s)
                    else:
                        break
                
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                "text": ' '.join(current_chunk),
                "length": len(' '.join(current_chunk))
            })
        
        self.trace.log_step("chunking", f"Semantic chunking complete",
                          {"total_chunks": len(chunks),
                           "chunk_size": chunk_size,
                           "overlap": overlap})
        
        return chunks
    
    def retrieve(self, query: str) -> Dict:
        """
        Retrieve documents using your existing ask.py functionality.
        Returns both the answer and source documents for verification.
        """
        self.trace.log_step("retrieval", f"Searching local database",
                          {"query": query})
        
        # Use your existing answer_question function
        answer = ask.answer_question(query)
        
        # Also get raw documents for verification
        results = ask.collection.query(
            query_texts=[query],
            n_results=5,
            include=["documents", "metadatas", "distances"]
        )
        
        retrieval_info = {
            "answer": answer,
            "documents": results['documents'][0] if results['documents'] else [],
            "metadatas": results['metadatas'][0] if results['metadatas'] else [],
            "count": len(results['documents'][0]) if results['documents'] else 0
        }
        
        self.trace.log_step("retrieval", f"Retrieved {retrieval_info['count']} documents")
        
        return retrieval_info


# ============================================================================
# MODULE 2: TOOL-CALLING MODULE
# ============================================================================

class ToolRegistry:
    """
    Tool registry that integrates your existing modules as callable tools.
    Implements the Tool-Calling Module requirement for HW2.
    """
    
    def __init__(self, trace: ExecutionTrace):
        self.trace = trace
        self.retriever = EnhancedRetriever(trace)
        
        # Define available tools using your existing modules
        self.tools = {
            "search_local_db": self.search_local_db,
            "search_arxiv": self.search_arxiv,
            "save_to_database": self.save_to_database,
            "rebuild_database": self.rebuild_database,
            "get_database_stats": self.get_database_stats
        }
        
        self.tool_descriptions = {
            "search_local_db": {
                "description": "Search the local ChromaDB for existing research papers and documents",
                "parameters": {"query": "string"},
                "use_when": "User asks about existing documents, saved papers, or 'what do I have'"
            },
            "search_arxiv": {
                "description": "Search ArXiv.org for new research papers (EXTERNAL API TOOL)",
                "parameters": {"topic": "string", "max_results": "integer"},
                "use_when": "User wants new papers, latest research, or unfamiliar topics"
            },
            "save_to_database": {
                "description": "Save ArXiv papers to the local database",
                "parameters": {"papers": "list"},
                "use_when": "User wants to save, store, or add papers"
            },
            "rebuild_database": {
                "description": "Rebuild database from PDF files in data folder",
                "parameters": {},
                "use_when": "User wants to re-ingest PDFs or refresh database"
            },
            "get_database_stats": {
                "description": "Get information about the database",
                "parameters": {},
                "use_when": "User asks about database contents or stats"
            }
        }
        
        self.trace.log_step("initialization", "Tool registry initialized",
                          {"available_tools": list(self.tools.keys())})
    
    def search_local_db(self, query: str) -> Dict:
        """Tool: Search local database using your ask.py module."""
        self.trace.log_step("tool_call", "Executing search_local_db",
                          {"query": query})
        
        try:
            result = self.retriever.retrieve(query)
            self.trace.log_step("tool_result", "Local DB search complete",
                              {"documents_found": result['count']})
            return result
        except Exception as e:
            error_msg = f"Error searching local DB: {str(e)}"
            self.trace.log_step("tool_error", error_msg)
            return {"answer": error_msg, "documents": [], "count": 0}
    
    def search_arxiv(self, topic: str, max_results: int = 5) -> Dict:
        """Tool: Search ArXiv using your arxiv_agent.py module."""
        self.trace.log_step("tool_call", "Executing search_arxiv (EXTERNAL API)",
                          {"topic": topic, "max_results": max_results})
        
        try:
            # Use your existing search_arxiv function
            papers = arxiv_agent.search_arxiv(topic, max_results)
            
            # Score papers using your existing function
            scored_papers = []
            for paper in papers:
                summary, score = arxiv_agent.summarize_and_score(paper, topic)
                scored_papers.append({
                    **paper,
                    "summary": summary,
                    "score": score
                })
            
            self.trace.log_step("tool_result", f"Found {len(papers)} papers on ArXiv",
                              {"topic": topic, "count": len(papers)})
            
            return {
                "papers": scored_papers,
                "count": len(scored_papers),
                "topic": topic
            }
        except Exception as e:
            error_msg = f"Error searching ArXiv: {str(e)}"
            self.trace.log_step("tool_error", error_msg)
            return {"papers": [], "count": 0, "error": error_msg}
    
    def save_to_database(self, papers: List[Dict]) -> str:
        """Tool: Save papers to database with semantic chunking."""
        self.trace.log_step("tool_call", "Executing save_to_database",
                          {"paper_count": len(papers)})
        
        try:
            saved_count = 0
            
            for paper in papers:
                # Create document text
                doc_text = f"TITLE: {paper['title']}\n\nABSTRACT: {paper.get('abstract', '')}"
                
                # Apply semantic chunking
                chunks = self.retriever.semantic_chunk(doc_text)
                
                # Save each chunk
                for i, chunk in enumerate(chunks):
                    paper_id = f"arxiv_{paper['link'].split('/')[-1]}_chunk{i}"
                    
                    self.retriever.collection.upsert(
                        documents=[chunk['text']],
                        metadatas=[{
                            "source": "arxiv",
                            "title": paper['title'],
                            "link": paper.get('link', ''),
                            "chunk_id": i,
                            "total_chunks": len(chunks),
                            "added_date": datetime.now().isoformat()
                        }],
                        ids=[paper_id]
                    )
                    saved_count += 1
            
            result_msg = f"Successfully saved {saved_count} chunks from {len(papers)} papers"
            self.trace.log_step("tool_result", result_msg)
            return result_msg
            
        except Exception as e:
            error_msg = f"Error saving to database: {str(e)}"
            self.trace.log_step("tool_error", error_msg)
            return error_msg
    
    def rebuild_database(self) -> str:
        """Tool: Rebuild database using your fill_db.py module."""
        self.trace.log_step("tool_call", "Executing rebuild_database")
        
        try:
            fill_db.fill()
            result_msg = "Database rebuilt successfully"
            self.trace.log_step("tool_result", result_msg)
            return result_msg
        except Exception as e:
            error_msg = f"Error rebuilding database: {str(e)}"
            self.trace.log_step("tool_error", error_msg)
            return error_msg
    
    def get_database_stats(self) -> Dict:
        """Tool: Get database statistics."""
        self.trace.log_step("tool_call", "Executing get_database_stats")
        
        try:
            count = self.retriever.collection.count()
            
            stats = {
                "total_documents": count,
                "collection_name": config.COLLECTION_NAME,
                "database_path": config.CHROMA_PATH
            }
            
            self.trace.log_step("tool_result", "Database stats retrieved", stats)
            return stats
            
        except Exception as e:
            error_msg = f"Error getting stats: {str(e)}"
            self.trace.log_step("tool_error", error_msg)
            return {"error": error_msg}
    
    def get_tool_descriptions(self) -> str:
        """Get formatted tool descriptions for the LLM."""
        descriptions = []
        for tool_name, info in self.tool_descriptions.items():
            desc = f"""
Tool: {tool_name}
Description: {info['description']}
Parameters: {info['parameters']}
Use when: {info['use_when']}
"""
            descriptions.append(desc)
        
        return "\n".join(descriptions)


# ============================================================================
# MODULE 3: VERIFICATION MODULE (GUARDRAILS)
# ============================================================================

class VerificationModule:
    """
    Hallucination detection and groundedness scoring.
    Implements the Verification Module requirement for HW2.
    """
    
    def __init__(self, trace: ExecutionTrace):
        self.trace = trace
    
    def calculate_groundedness_score(self, response: str, 
                                    source_documents: List[str]) -> Tuple[float, List[str]]:
        """
        Calculate Evidence Support Score (Groundedness Score).
        
        Returns:
            - score (0.0 to 1.0): Fraction of claims supported by sources
            - unsupported_claims: List of claims lacking evidence
        """
        self.trace.log_step("verification", "Calculating groundedness score")
        
        if not source_documents:
            self.trace.log_step("verification", "No source documents provided", 
                              {"score": 0.0})
            return 0.0, ["No source documents available"]
        
        # Extract factual claims from the response
        claims_prompt = f"""
Extract the main factual claims from this response. 
List each claim on a new line, numbered.
Only extract factual claims, not opinions or greetings.

Response:
{response}

Claims:"""
        
        claims_text = _call_llm("", claims_prompt)
        claims = [c.strip() for c in claims_text.split('\n') if c.strip()]
        claims = [re.sub(r'^\d+[\.\)]\s*', '', c) for c in claims]  # Remove numbering
        claims = [c for c in claims if len(c) > 10]  # Filter out very short items
        
        if not claims:
            self.trace.log_step("verification", "No factual claims extracted",
                              {"score": 1.0})
            return 1.0, []
        
        # Check each claim against source documents
        supported_count = 0
        unsupported_claims = []
        
        combined_sources = "\n\n".join(source_documents[:3])  # Limit to first 3 sources
        
        for claim in claims:
            verification_prompt = f"""
Is the following claim supported by the source documents?

Claim: {claim}

Source Documents:
{combined_sources[:2000]}

Answer with only: YES or NO"""
            
            verification = _call_llm("", verification_prompt).strip().upper()
            
            if "YES" in verification:
                supported_count += 1
            else:
                unsupported_claims.append(claim)
        
        score = supported_count / len(claims) if claims else 1.0
        
        self.trace.log_step("verification", f"Groundedness score calculated: {score:.2f}",
                          {"total_claims": len(claims),
                           "supported": supported_count,
                           "unsupported": len(unsupported_claims)})
        
        return score, unsupported_claims
    
    def verify_response(self, response: str, source_documents: List[str],
                       confidence_threshold: float = 0.7) -> Dict:
        """
        Self-evaluation node that verifies response quality.
        
        Returns verification results including groundedness score and recommendations.
        """
        score, unsupported = self.calculate_groundedness_score(response, source_documents)
        
        verification_result = {
            "groundedness_score": score,
            "passed": score >= confidence_threshold,
            "unsupported_claims": unsupported,
            "threshold": confidence_threshold,
            "recommendation": ""
        }
        
        if score >= confidence_threshold:
            verification_result["recommendation"] = "Response is well-grounded in source material"
        elif score >= 0.5:
            verification_result["recommendation"] = "Response is partially grounded. Consider requesting more specific sources"
        else:
            verification_result["recommendation"] = "Response may contain hallucinations. Seek clarification or additional sources"
        
        self.trace.log_step("verification", "Response verification complete",
                          verification_result)
        
        return verification_result


# ============================================================================
# MAIN AGENT: ReAct-STYLE REASONING LOOP
# ============================================================================

class AutonomousResearchAgent:
    """
    Main agentic system with ReAct reasoning loop.
    Integrates all your existing modules (ask.py, arxiv_agent.py, fill_db.py).
    """
    
    def __init__(self):
        self.trace = ExecutionTrace()
        self.tools = ToolRegistry(self.trace)
        self.verifier = VerificationModule(self.trace)
        self.conversation_history = []
        self.last_arxiv_results = []
        
        self.trace.log_step("initialization", "Autonomous Research Agent initialized")
        print("\n✅ Agent ready! Using your existing modules:")
        print("   - ask.py (local retrieval)")
        print("   - arxiv_agent.py (ArXiv API)")
        print("   - fill_db.py (database management)")
        print("   - llm_clients.py (LLM providers)")
    
    def reason_and_act(self, user_query: str) -> Dict:
        """
        ReAct loop: Reason → Act → Observe → Verify → Respond
        """
        self.trace.log_step("user_query", user_query, {"query": user_query})
        
        # STEP 1: REASONING
        reasoning_result = self._reasoning_step(user_query)
        
        # STEP 2: ACTION
        action_result = self._action_step(reasoning_result)
        
        # STEP 3: GENERATE RESPONSE
        response = self._generate_response(user_query, action_result)
        
        # STEP 4: VERIFICATION
        source_docs = action_result.get("source_documents", [])
        verification = self.verifier.verify_response(response, source_docs)
        
        # Compile final result
        final_result = {
            "query": user_query,
            "reasoning": reasoning_result,
            "action_taken": action_result,
            "response": response,
            "verification": verification,
            "timestamp": datetime.now().isoformat()
        }
        
        self.conversation_history.append(final_result)
        
        return final_result
    
    def _reasoning_step(self, query: str) -> Dict:
        """REASONING: Decide what action to take."""
        self.trace.log_step("reasoning", "Determining best action for query")
        
        recent_context = self._get_recent_context()
        
        reasoning_prompt = f"""
You are a research assistant agent. Analyze the user's query and decide the best action.

User Query: "{query}"

Available Tools:
{self.tools.get_tool_descriptions()}

Recent Context:
{recent_context}

Reasoning Instructions:
1. If query is about EXISTING documents or saved papers → use search_local_db
2. If query asks for NEW research or unfamiliar topics → use search_arxiv
3. If query asks to SAVE papers (and papers were just found) → use save_to_database
4. If query asks to REBUILD database or re-ingest PDFs → use rebuild_database
5. If query asks about DATABASE INFO → use get_database_stats
6. If unclear → use clarify

Respond in this EXACT format:
THOUGHT: [Your reasoning about what the user needs]
ACTION: [Tool name: search_local_db, search_arxiv, save_to_database, rebuild_database, get_database_stats, or clarify]
PARAMETERS: [JSON object with parameters]
"""
        
        reasoning_output = _call_llm("You are a reasoning agent", reasoning_prompt)
        parsed = self._parse_reasoning(reasoning_output)
        
        self.trace.log_step("reasoning", f"Decision: {parsed['action']}",
                          {"thought": parsed['thought'][:100] + "...",
                           "parameters": parsed['parameters']})
        
        return parsed
    
    def _parse_reasoning(self, reasoning_output: str) -> Dict:
        """Parse the ReAct-style reasoning output."""
        thought_match = re.search(r'THOUGHT:\s*(.+?)(?=ACTION:|$)', reasoning_output, re.DOTALL)
        action_match = re.search(r'ACTION:\s*(\w+)', reasoning_output)
        params_match = re.search(r'PARAMETERS:\s*({.+})', reasoning_output, re.DOTALL)
        
        thought = thought_match.group(1).strip() if thought_match else "No explicit reasoning provided"
        action = action_match.group(1).strip() if action_match else "clarify"
        
        try:
            parameters = json.loads(params_match.group(1)) if params_match else {}
        except:
            parameters = {}
        
        return {
            "thought": thought,
            "action": action,
            "parameters": parameters
        }
    
    def _action_step(self, reasoning: Dict) -> Dict:
        """ACTION: Execute the chosen tool."""
        action = reasoning['action']
        parameters = reasoning['parameters']
        
        self.trace.log_step("action", f"Executing action: {action}", parameters)
        
        result = {
            "action": action,
            "parameters": parameters,
            "output": None,
            "source_documents": []
        }
        
        if action == "search_local_db":
            query = parameters.get("query", "")
            retrieval_result = self.tools.search_local_db(query)
            result["output"] = retrieval_result['answer']
            result["source_documents"] = retrieval_result['documents']
            
        elif action == "search_arxiv":
            topic = parameters.get("topic", "")
            max_results = parameters.get("max_results", 5)
            arxiv_result = self.tools.search_arxiv(topic, max_results)
            result["output"] = arxiv_result['papers']
            result["source_documents"] = [f"{p['title']}: {p.get('abstract', '')[:200]}" 
                                        for p in arxiv_result['papers']]
            # Store for potential save operation
            self.last_arxiv_results = arxiv_result['papers']
            
        elif action == "save_to_database":
            papers = parameters.get("papers", self.last_arxiv_results)
            if papers:
                output_msg = self.tools.save_to_database(papers)
                result["output"] = output_msg
                self.last_arxiv_results = []  # Clear after saving
            else:
                result["output"] = "No papers to save. Please search ArXiv first."
            
        elif action == "rebuild_database":
            output_msg = self.tools.rebuild_database()
            result["output"] = output_msg
            
        elif action == "get_database_stats":
            stats = self.tools.get_database_stats()
            result["output"] = stats
            
        else:  # clarify
            result["output"] = "I need more information. Could you please clarify your request?"
        
        return result
    
    def _generate_response(self, query: str, action_result: Dict) -> str:
        """Generate natural language response from action results."""
        self.trace.log_step("response_generation", "Generating user-facing response")
        
        output_str = json.dumps(action_result['output'], indent=2)[:1500] if action_result['output'] else "No results"
        
        generation_prompt = f"""
Generate a helpful, natural response to the user's query based on the information gathered.

User Query: {query}

Action Taken: {action_result['action']}
Information Gathered:
{output_str}

Instructions:
- Be concise and directly answer the query
- Cite sources when available
- If multiple papers found, summarize key findings
- Be honest about limitations

Response:"""
        
        response = _call_llm("You are a helpful research assistant", generation_prompt)
        
        return response.strip()
    
    def _get_recent_context(self) -> str:
        """Get recent conversation context."""
        if not self.conversation_history:
            return "No previous context"
        
        recent = self.conversation_history[-2:]
        context_summary = []
        for item in recent:
            context_summary.append(f"Previous: {item['query']} | Action: {item['action_taken']['action']}")
        
        if self.last_arxiv_results:
            context_summary.append(f"Note: {len(self.last_arxiv_results)} ArXiv papers available from last search")
        
        return "\n".join(context_summary)
    
    def print_result(self, result: Dict):
        """Pretty print the agent's result."""
        print("\n" + "="*80)
        print("🤖 AGENT EXECUTION RESULT")
        print("="*80)
        
        print(f"\n📝 Query: {result['query']}")
        
        print(f"\n💭 Reasoning:")
        print(f"   {result['reasoning']['thought'][:150]}...")
        print(f"   → Action: {result['reasoning']['action']}")
        
        print(f"\n🎯 Response:")
        print(f"   {result['response']}")
        
        print(f"\n✅ Verification (Groundedness Score):")
        print(f"   Score: {result['verification']['groundedness_score']:.2f}")
        print(f"   Status: {'✓ PASSED' if result['verification']['passed'] else '✗ FAILED'}")
        print(f"   {result['verification']['recommendation']}")
        
        if result['verification']['unsupported_claims']:
            print(f"\n⚠️  Unsupported Claims:")
            for claim in result['verification']['unsupported_claims'][:3]:
                print(f"   • {claim}")
        
        print("\n" + "="*80)
    
    def save_execution_trace(self, filename: str = "execution_trace.json"):
        """Save the execution trace to file."""
        return self.trace.save_trace(filename)


# ============================================================================
# MAIN INTERFACE
# ============================================================================

def main():
    """Main interface for the autonomous agent."""
    print("="*80)
    print("  🤖 AUTONOMOUS RESEARCH AGENT - HW2")
    print("  Integrates with your existing modules")
    print("="*80)
    print("\nFeatures:")
    print("  ✓ ReAct reasoning loop")
    print("  ✓ Semantic chunking (500-char with 100-char overlap)")
    print("  ✓ Tool calling (ArXiv API, Local DB)")
    print("  ✓ Groundedness scoring & hallucination detection")
    print("\nCommands:")
    print("  'exit' - Quit and save trace")
    print("  'trace' - Save execution trace now")
    print("  'stats' - Show database stats")
    print("\n" + "="*80)
    
    agent = AutonomousResearchAgent()
    
    while True:
        print("\n")
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['exit', 'quit']:
            print("\n👋 Saving execution trace and exiting...")
            agent.save_execution_trace()
            break
        
        if user_input.lower() == 'trace':
            agent.save_execution_trace()
            continue
        
        if user_input.lower() == 'stats':
            stats = agent.tools.get_database_stats()
            print(f"\n📊 Database Stats:")
            print(f"   Total documents: {stats['total_documents']}")
            print(f"   Collection: {stats['collection_name']}")
            continue
        
        # Process query through agent
        try:
            result = agent.reason_and_act(user_input)
            agent.print_result(result)
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            agent.trace.log_step("error", str(e))


if __name__ == "__main__":
    main()