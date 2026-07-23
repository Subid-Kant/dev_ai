import os
import json
import re

def build_dataset_jsonl(knowledge_dir="knowledge", output_file="data/dataset.jsonl"):
    """
    Parses Q&A pairs from text files and formats them into JSONL for HuggingFace Instruction tuning.
    Expected format: Llama-3 conversational chat style.
    """
    print(f"Building Instruction Dataset from {knowledge_dir}...")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    files = [f for f in os.listdir(knowledge_dir) if f.endswith('.txt')]
    
    dataset_entries = []
    
    for filename in files:
        topic = os.path.splitext(filename)[0]
        with open(os.path.join(knowledge_dir, filename), "r", encoding="utf-8") as f:
            content = f.read()
            
        # Very basic parser: assume lines starting with "Q: " or "?" are questions,
        # and subsequent lines are the answer. If the file is just raw text, chunk it.
        # We will split by double newlines to find paragraphs, then frame them as queries.
        
        # We will parse our specific strict format if available, otherwise chunk.
        blocks = re.split(r'\n\s*\n', content.strip())
        
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            
            # Simple heuristic for Q&A vs Raw knowledge
            if block.startswith("Q:") or block.startswith("Question:"):
                parts = re.split(r'\nA:|\nAnswer:', block, maxsplit=1)
                if len(parts) == 2:
                    q = parts[0].replace("Q:", "").replace("Question:", "").strip()
                    a = parts[1].strip()
                    dataset_entries.append({"messages": [
                        {"role": "system", "content": f"You are a master developer AI assistant. Focus on {topic}."},
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a}
                    ]})
                    continue

            # Fallback: Turn raw informational block into a completion task
            query = f"Explain the following concept regarding {topic}."
            dataset_entries.append({"messages": [
                {"role": "system", "content": f"You are a master developer AI assistant specializing in {topic}."},
                {"role": "user", "content": query},
                {"role": "assistant", "content": block}
            ]})
            
    # Add Synthetic Tool-Use and RAG examples to train the reasoning model
    synthetic_rag_examples = [
        {"messages": [
            {"role": "system", "content": "You are a master AI. You have access to a Vector Database to search for real-time information."},
            {"role": "user", "content": "Search the database for the latest news on Quantum Computing."},
            {"role": "assistant", "content": "I will search the Vector Database for recent news on Quantum Computing.\n<tool_call>\n{\"name\": \"search_vector_db\", \"arguments\": {\"query\": \"latest news quantum computing\"}}\n</tool_call>"}
        ]},
        {"messages": [
            {"role": "system", "content": "You are a master AI. You have access to a Vector Database to search for real-time information."},
            {"role": "user", "content": "Here are the search results: [Document 1: IBM announces new 1000 qubit chip...]. What is the latest on quantum computing?"},
            {"role": "assistant", "content": "Based on the latest search results from the Vector Database, IBM has just announced a new 1000 qubit quantum computing chip."}
        ]}
    ]
    dataset_entries.extend(synthetic_rag_examples)

    # Save to JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in dataset_entries:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Successfully generated {output_file} with {len(dataset_entries)} conversation turns.")

if __name__ == "__main__":
    build_dataset_jsonl()
