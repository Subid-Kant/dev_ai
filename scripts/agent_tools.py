import os
from duckduckgo_search import DDGS
import chromadb

def search_vector_db(query, n_results=3):
    """
    Search the local/cloud ChromaDB instance where the continuous crawler stores its knowledge.
    This acts as the 'Memory' of what the AI has learned in the background.
    """
    db_path = os.environ.get("CHROMA_DB_PATH", "./data/chroma_db")
    if not os.path.exists(db_path):
        return "Vector Database not found. The background crawler may not have run yet."
        
    client = chromadb.PersistentClient(path=db_path)
    try:
        collection = client.get_collection(name="world_knowledge")
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        if not results['documents'][0]:
            return "No relevant background knowledge found in the Vector Database."
            
        formatted_results = []
        for i, doc in enumerate(results['documents'][0]):
            meta = results['metadatas'][0][i] if results['metadatas'] else {}
            formatted_results.append(f"[{meta.get('source', 'Unknown')}]: {doc}")
            
        return "\n\n".join(formatted_results)
    except Exception as e:
        return f"Error querying Vector Database: {e}"

def live_web_search(query, max_results=3):
    """
    Perform a live web search if the Vector DB doesn't have the answer.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            if not results:
                return "No web search results found."
                
            formatted = []
            for res in results:
                formatted.append(f"Title: {res.get('title')}\nSnippet: {res.get('body')}\nURL: {res.get('href')}")
            return "\n\n".join(formatted)
    except Exception as e:
        return f"Web search failed: {e}"

# Tool registry to map tool names to functions for the LLM
TOOL_REGISTRY = {
    "search_vector_db": search_vector_db,
    "live_web_search": live_web_search
}
