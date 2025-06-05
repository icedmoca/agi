from core.memory import Memory
from core.vector_memory import get_vectorizer
import json
from pathlib import Path

def test_vector_memory():
    # Initialize memory with vectorizer
    memory = Memory(filename="memory.jsonl", vectorizer=get_vectorizer())
    
    # Add some test entries if memory is empty
    if not memory.entries:
        test_entries = [
            ("Fix bug in login system", "Fixed authentication token validation"),
            ("Update user profile page", "Added new fields and validation"),
            ("Optimize database queries", "Added indexes and improved joins"),
            ("Implement file upload", "Added multipart form handling"),
            ("Add error logging", "Integrated with logging service")
        ]
        
        for goal, result in test_entries:
            memory.append(goal=goal, result=result)
    
    # Interactive search loop
    while True:
        query = input("\n🔍 Enter a goal to search memory for (or 'q' to quit): ")
        if query.lower() == 'q':
            break
            
        similar = memory.find_similar(query, top_k=3)
        
        if not similar:
            print("No similar entries found.")
        else:
            print("\nTop similar memory entries:")
            for entry in similar:
                print(f"• {entry['goal']}: {entry['result']}")

if __name__ == "__main__":
    test_vector_memory() 