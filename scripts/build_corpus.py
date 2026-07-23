import os
import json
import re
import hashlib

def build_dataset_jsonl(knowledge_dir="knowledge", output_file="data/dataset.jsonl"):
    """
    Parses Q&A pairs from text files and formats them into JSONL for HuggingFace Instruction tuning.
    Expected format: Llama-3 conversational chat style.
    """
    print(f"Building Instruction Dataset from {knowledge_dir}...")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if not os.path.isdir(knowledge_dir):
        print(f"'{knowledge_dir}' does not exist yet (crawler hasn't produced any files). "
              f"Writing an empty dataset so downstream steps don't crash.")
        open(output_file, "w", encoding="utf-8").close()
        return

    files = [f for f in os.listdir(knowledge_dir) if f.endswith(".txt")]

    dataset_entries = []
    seen_hashes = set()  # avoid duplicate training examples if a block appears twice

    def add_entry(entry):
        key = hashlib.sha256(json.dumps(entry, sort_keys=True).encode("utf-8")).hexdigest()
        if key not in seen_hashes:
            seen_hashes.add(key)
            dataset_entries.append(entry)

    for filename in files:
        topic = os.path.splitext(filename)[0]
        with open(os.path.join(knowledge_dir, filename), "r", encoding="utf-8") as f:
            content = f.read()

        # Blocks separated by a blank line. The crawler writes blocks as:
        #   Source: <url>
        #   Title: <title>
        #   <content>
        blocks = re.split(r"\n\s*\n", content.strip())

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            # Q&A style files (manually authored), unrelated to the crawler output
            if block.startswith("Q:") or block.startswith("Question:"):
                parts = re.split(r"\nA:|\nAnswer:", block, maxsplit=1)
                if len(parts) == 2:
                    q = parts[0].replace("Q:", "").replace("Question:", "").strip()
                    a = parts[1].strip()
                    add_entry({"messages": [
                        {"role": "system", "content": f"You are a master developer AI assistant. Focus on {topic}."},
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a}
                    ]})
                    continue

            # Crawler-produced blocks: strip the "Source:"/"Title:" header lines
            # before using the remainder as the assistant's answer content.
            lines = block.split("\n")
            source_line = next((l for l in lines if l.startswith("Source:")), None)
            title_line = next((l for l in lines if l.startswith("Title:")), None)
            body_lines = [
                l for l in lines
                if not l.startswith("Source:") and not l.startswith("Title:")
            ]
            body = "\n".join(body_lines).strip()
            if not body:
                continue

            query = f"Tell me what you know about {topic}."
            if title_line:
                query = f"Tell me about: {title_line.replace('Title:', '').strip()}"

            add_entry({"messages": [
                {"role": "system", "content": f"You are a master developer AI assistant specializing in {topic}."},
                {"role": "user", "content": query},
                {"role": "assistant", "content": body}
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
    for entry in synthetic_rag_examples:
        add_entry(entry)

    with open(output_file, "w", encoding="utf-8") as f:
        for entry in dataset_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"Successfully generated {output_file} with {len(dataset_entries)} conversation turns.")

if __name__ == "__main__":
    build_dataset_jsonl()