import time
import os
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import chromadb
from chromadb.config import Settings

# Initialize ChromaDB (can be configured to use a remote instance or local persistent dir)
# For free tier cloud deployment, a local sqlite persistent dir is often easiest,
# or connecting to a free tier cloud vector DB like Pinecone/Supabase if preferred.
# Here we use Chroma with persistent storage which is easy to deploy on Render (with a disk)
# or PythonAnywhere.
DB_PATH = os.environ.get("CHROMA_DB_PATH", "./data/chroma_db")
client = chromadb.PersistentClient(path=DB_PATH)

# Using a lightweight sentence-transformer model that can run without GPU
collection = client.get_or_create_collection(name="world_knowledge")

def fetch_and_parse_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract main text (very basic heuristic)
        paragraphs = soup.find_all('p')
        text = "\n".join([p.get_text() for p in paragraphs if len(p.get_text()) > 50])
        return text
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def continuous_learning_cycle():
    """
    A cycle that queries the web for random/trending topics and ingests them into the DB.
    """
    print("Starting a new learning cycle...")
    topics = [
        "latest artificial intelligence breakthroughs",
        "new programming languages 2026",
        "quantum computing advancements",
        "global technology news",
        "cloud computing architecture best practices"
    ]
    
    with DDGS() as ddgs:
        for topic in topics:
            print(f"Researching topic: {topic}")
            results = list(ddgs.text(topic, max_results=3))
            
            for res in results:
                url = res.get('href')
                title = res.get('title', 'Unknown Title')
                body_snippet = res.get('body', '')
                
                print(f" -> Learning from: {title} ({url})")
                
                # We could fetch the full URL, but for speed and stability, we use the snippet first
                # full_text = fetch_and_parse_url(url)
                # content = full_text if full_text else body_snippet
                content = body_snippet
                
                if content:
                    doc_id = url
                    collection.add(
                        documents=[f"Title: {title}\nContent: {content}"],
                        metadatas=[{"source": url, "topic": topic}],
                        ids=[doc_id]
                    )
                    print(f"    [+] Ingested to Vector DB: {doc_id}")
            time.sleep(2) # Respect rate limits

if __name__ == "__main__":
    print("Cloud Crawler background agent started via GitHub Actions.")
    try:
        continuous_learning_cycle()
        print("Cycle complete. Database updated successfully.")
    except Exception as e:
        print(f"Agent encountered an error: {e}")
