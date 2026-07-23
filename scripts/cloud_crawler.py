import time
import os
import re
import json
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import chromadb

# --- Config ---
DB_PATH = os.environ.get("CHROMA_DB_PATH", "./data/chroma_db")
KNOWLEDGE_DIR = os.environ.get("KNOWLEDGE_DIR", "./knowledge")
MANIFEST_PATH = os.environ.get("CRAWL_MANIFEST", "./data/crawled_urls.json")

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name="world_knowledge")

os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)


def load_manifest():
    """Set of URLs we've already ingested, so re-running every 6 hours doesn't
    just re-add the same handful of search results forever."""
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            try:
                return set(json.load(f))
            except json.JSONDecodeError:
                return set()
    return set()


def save_manifest(seen_urls):
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted(seen_urls), f)


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:60]


def fetch_and_parse_url(url):
    try:
        response = requests.get(
            url, timeout=10, headers={"User-Agent": "Mozilla/5.0 (compatible; DevAI-Crawler/1.0)"}
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = "\n".join(p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 50)
        return text.strip() or None
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None


def append_to_knowledge_file(topic, title, url, content):
    """Write real content into knowledge/<topic>.txt in the raw-text block
    format build_corpus.py expects (blocks separated by a blank line)."""
    path = os.path.join(KNOWLEDGE_DIR, f"{slugify(topic)}.txt")
    block = f"Source: {url}\nTitle: {title}\n{content}\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(block + "\n\n")


def continuous_learning_cycle():
    print("Starting a new learning cycle...")
    topics = [
        "latest artificial intelligence breakthroughs",
        "new programming languages 2026",
        "quantum computing advancements",
        "global technology news",
        "cloud computing architecture best practices",
    ]

    seen_urls = load_manifest()
    new_count = 0

    with DDGS() as ddgs:
        for topic in topics:
            print(f"Researching topic: {topic}")
            results = list(ddgs.text(topic, max_results=5))

            for res in results:
                url = res.get("href")
                title = res.get("title", "Unknown Title")
                body_snippet = res.get("body", "")

                if not url or url in seen_urls:
                    continue  # skip URLs we've already ingested in a prior run

                print(f" -> Learning from: {title} ({url})")

                full_text = fetch_and_parse_url(url)
                content = full_text if full_text else body_snippet

                if content:
                    collection.add(
                        documents=[f"Title: {title}\nContent: {content}"],
                        metadatas=[{"source": url, "topic": topic}],
                        ids=[url],
                    )
                    append_to_knowledge_file(topic, title, url, content)
                    seen_urls.add(url)
                    new_count += 1
                    print(f"    [+] Ingested (Chroma + knowledge/*.txt): {url}")

            time.sleep(2)  # Respect rate limits

    save_manifest(seen_urls)
    print(f"Cycle finished. {new_count} new documents ingested this run.")


if __name__ == "__main__":
    print("Cloud Crawler background agent started via GitHub Actions.")
    try:
        continuous_learning_cycle()
        print("Cycle complete. Database updated successfully.")
    except Exception as e:
        print(f"Agent encountered an error: {e}")