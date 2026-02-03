import sys
import ask          # Refactored local RAG module
import arxiv_agent  # Refactored online research module
import fill_db      # Refactored database ingestion module

def run_agent():
    print("==========================================")
    print("   🌿 AI RESEARCH ASSISTANT   ")
    print("==========================================")
    
    while True:
        print("\nMAIN MENU")
        print("1. Ask Local Database ")
        print("2. Search Online Papers (ArXiv)")
        print("3. Re-build Database (Ingest new PDFs)")
        print("4. Exit")
        print("------------------------------------------")
        
        option = input("Select an option (1-4): ").strip()

        # OPTION 1: Local RAG (Ask questions to your PDFs)
        if option == "1":
            print("\n--- Local Database Mode ---")
            print("Type 'back' to return to the main menu.")
            
            while True:
                query = input("\nWhat do you want to know about from your existing data\n(or 'back'): ").strip()
                
                if query.lower() == 'back':
                    break
                
                if query:
                    # Calls the answer_question function from ask.py
                    answer = ask.answer_question(query)
                    print("\n🤖 AI Response:\n")
                    print(answer)
                    print("-" * 40)
                else:
                    print("Please enter a valid question.")

        # OPTION 2: Online Research (Search ArXiv)
        elif option == "2":
            print("\n--- 🌐 Online Research Mode ---")
            topic = input("\nEnter a research topic:\n> ").strip()
            
            if not topic:
                print("Topic cannot be empty.")
                continue

            print(f"\n🔍 Searching arXiv for '{topic}'...")
            
            # Calls search_arxiv from arxiv_agent.py
            papers = arxiv_agent.search_arxiv(topic)
            
            if not papers:
                print("No papers found.")
            else:
                print(f"Found {len(papers)} papers. analyzing relevance...")
                
                count = 0
                for i, paper in enumerate(papers, 1):
                    # Calls summarize_and_score from arxiv_agent.py
                    summary, score = arxiv_agent.summarize_and_score(paper, topic)
                    
                    if score >= 2:  # Filter: Only show relevant papers
                        count += 1
                        print(f"\n📄 Paper {i}: {paper['title']}")
                        print(f"   Relevance Score: {score}/3")
                        print(f"   Link: {paper['link']}")
                        print("\n   Summary:")
                        print(summary)
                        print("-" * 60)
                
                if count == 0:
                    print("Papers were found, but none were relevant enough (Score < 2).")

        # OPTION 3: Database Maintenance
        elif option == "3":
            print("\n--- 💾 Database Maintenance ---")
            confirm = input("This will delete the current database and reload from the 'data' folder.\nAre you sure? (y/n): ").lower()
            if confirm == 'y':
                # Calls the fill function from fill_db.py
                fill_db.fill()
            else:
                print("Operation cancelled.")

        # OPTION 4: Exit
        elif option == "4":
            print("Goodbye! ")
            sys.exit() # Clean exit
        
        else:
            print("\n❌ Invalid option. Please try again.")

if __name__ == "__main__":
    # Ensure this script is the entry point
    try:
        run_agent()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Exiting...")
        sys.exit()