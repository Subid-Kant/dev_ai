"""
retrieve.py

This is the piece that was missing entirely from the original setup: something
that actually *reads* from the Chroma DB when answering a question. Without
this, crawling and storing documents does nothing at inference time -- the DB
just sits there.

Usage as a library:
    from retrieve import retrieve_context
    context = retrieve_context("what's new in quantum computing?", n_results=3)

Usage standalone:
    python retrieve.py "what's new in quantum computing?"
"""
import os
import sys
import chromadb

DB_PATH = os.environ.get("CHROMA_DB_PATH", "./data/chroma_db")

_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=DB_PATH)
        _collection = _client.get_or_create_collection(name="world_knowledge")
    return _collection


def retrieve_context(query, n_results=3):
    """Return a list of {document, source, topic} dicts most relevant to query."""
    collection = _get_collection()
    if collection.count() == 0:
        return []

    results = collection.query(query_texts=[query], n_results=min(n_results, collection.count()))

    hits = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    for doc, meta in zip(documents, metadatas):
        hits.append({
            "document": doc,
            "source": (meta or {}).get("source", "unknown"),
            "topic": (meta or {}).get("topic", "unknown"),
        })
    return hits


def build_rag_prompt(user_question, n_results=3):
    """Assemble a prompt block you can feed to your LLM: retrieved context + the
    user's actual question. This is the standard RAG pattern -- retrieval
    happens here, the LLM never touches the DB directly."""
    hits = retrieve_context(user_question, n_results=n_results)

    if not hits:
        return user_question  # nothing in the DB yet, fall back to plain question

    context_block = "\n\n".join(
        f"[Source: {h['source']}]\n{h['document']}" for h in hits
    )
    prompt = (
        "Use the following retrieved context if relevant. If it doesn't help, "
        "answer from your own knowledge instead.\n\n"
        f"--- Retrieved context ---\n{context_block}\n--- End context ---\n\n"
        f"Question: {user_question}"
    )
    return prompt


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "latest artificial intelligence breakthroughs"
    print(f"Query: {q}\n")
    for i, hit in enumerate(retrieve_context(q, n_results=3), 1):
        print(f"[{i}] ({hit['topic']}) {hit['source']}")
        print(hit["document"][:300] + ("..." if len(hit["document"]) > 300 else ""))
        print()