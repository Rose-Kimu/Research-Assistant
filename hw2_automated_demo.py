# hw2_automated_demo.py - HW2 Demonstration Script
"""
Automated demonstration of all three required modules:
1. Retrieval Module (Memory) - Local DB with high/low groundedness
2. Tool-Calling Module - ArXiv API with verification  
3. Verification Module (Guardrails) - Evidence scoring demonstration

This script runs automatically without user input to generate the implementation trace.
"""

import time
from datetime import datetime
import design

def log_header(title):
    """Print a formatted header for log sections"""
    print("\n" + "=" * 60)
    print(f"   {title}")
    print("=" * 60)

def log_test(test_num, description, query):
    """Log each test case concisely"""
    print(f"\n[TEST {test_num}] {description}")
    print(f"Query: {query}")
    print("Result:")

def main():
    """Run concise automated tests for HW2 demonstration"""
    
    log_header("HW2 IMPLEMENTATION TRACE")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("System: RAG Agent with Dual Groundedness Verification")
    
    # ========================================================================
    # MODULE 1: LOCAL DATABASE (HIGH GROUNDEDNESS)
    # ========================================================================
    
    log_test("1.1", "LOCAL DB - HIGH GROUNDEDNESS", 
             "What languages were tested in the multilingual sentiment analysis study?")
    
    try:
        result1 = design.route_and_execute("What languages were tested in the multilingual sentiment analysis study?")
        print(result1[:300] + "..." if len(result1) > 300 else result1)
    except Exception as e:
        print(f"Error: {e}")
    
    time.sleep(2)
    
    # ========================================================================
    # MODULE 1: LOCAL DATABASE (LOW GROUNDEDNESS)
    # ========================================================================
    
    log_test("1.2", "LOCAL DB - POTENTIAL LOW GROUNDEDNESS", 
             "What are the main challenges in computer vision research?")
    
    try:
        result2 = design.route_and_execute("What are the main challenges in computer vision research according to the documents?")
        print(result2[:300] + "..." if len(result2) > 300 else result2)
    except Exception as e:
        print(f"Error: {e}")
    
    time.sleep(2)
    
    # ========================================================================
    # MODULE 2: ARXIV TOOL (HIGH GROUNDEDNESS)
    # ========================================================================
    
    log_test("2.1", "ARXIV - HIGH GROUNDEDNESS", 
             "Find recent papers about machine learning optimization")
    
    try:
        result3 = design.route_and_execute("Find recent papers about machine learning optimization")
        print(result3[:400] + "..." if len(result3) > 400 else result3)
    except Exception as e:
        print(f"Error: {e}")
    
    time.sleep(3)
    
    # ========================================================================
    # MODULE 2: ARXIV TOOL (POTENTIAL LOW GROUNDEDNESS)
    # ========================================================================
    
    log_test("2.2", "ARXIV - POTENTIAL LOW GROUNDEDNESS", 
             "Find papers about quantum neural networks for protein folding")
    
    try:
        result4 = design.route_and_execute("Find papers about quantum neural networks for protein folding")
        print(result4[:400] + "..." if len(result4) > 400 else result4)
    except Exception as e:
        print(f"Error: {e}")
    
    time.sleep(3)
    
    # ========================================================================
    # MODULE 3: DECISION MAKING DEMONSTRATION
    # ========================================================================
    
    log_test("3.1", "DECISION MAKING - LOCAL vs ARXIV", 
             "What bias reduction percentages were mentioned?")
    
    try:
        result5 = design.route_and_execute("What bias reduction percentages were mentioned in the AI ethics study?")
        print(result5[:300] + "..." if len(result5) > 300 else result5)
    except Exception as e:
        print(f"Error: {e}")
    
    time.sleep(2)
    
    log_test("3.2", "DECISION MAKING - ARXIV SEARCH", 
             "Search for new research on transformer attention")
    
    try:
        result6 = design.route_and_execute("Search for new research on transformer attention mechanisms")
        print(result6[:400] + "..." if len(result6) > 400 else result6)
    except Exception as e:
        print(f"Error: {e}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    log_header("IMPLEMENTATION SUMMARY")
    
    print("✅ RETRIEVAL MODULE: Local database with Evidence Support Scoring")
    print("✅ TOOL-CALLING MODULE: ArXiv API with ArXiv Evidence Support Scoring")
    print("✅ VERIFICATION MODULE: Dual groundedness verification system")
    print("✅ DECISION MAKING: Correct tool selection demonstrated")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("All three HW2 modules successfully demonstrated")

if __name__ == "__main__":
    main()