"""
BM25 Retriever — loads knowledge base files, chunks them, builds BM25 index.
Used for RAG: retrieve relevant context before LLM generation.
"""

import os
import re
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Chunk:
    id: str
    topic: str
    title: str
    text: str
    score: float = 0.0


STOP_WORDS = {
    "a","an","the","is","are","was","were","be","been","do","does","did",
    "will","would","could","should","in","on","at","to","for","of","with",
    "by","from","and","or","but","not","this","that","i","you","we","it",
    "can","use","how","what","why","when","where","which","who",
}

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "python": ["python","py","pip","django","flask","fastapi","numpy","pandas",
               "pytest","asyncio","pydantic","decorator","generator","coroutine",
               "typing","mypy","ruff","uvicorn","virtualenv","venv"],
    "android": ["android","kotlin","gradle","jetpack","compose","viewmodel",
                "coroutine","room","retrofit","hilt","flow","stateflow","shareflow",
                "recyclerview","fragment","activity","intent","manifest","adb"],
    "ios": ["ios","swift","swiftui","xcode","combine","coredata","urlsession",
            "optional","guard","actor","async","await","swiftdata","objc",
            "storyboard","autolayout","uikit","viewcontroller","cocoapods"],
    "windows": ["windows","csharp","c#","dotnet",".net","wpf","winforms",
               "winui","xaml","aspnet","entity","efcore","powershell","nuget",
               "blazor","signalr","msix","uwp","visualstudio","linq"],
    "mac": ["mac","macos","cocoa","objective-c","swift","appkit","xcode","dmg","pkg"],
    "fullstack": ["fullstack","frontend","backend","react","node","express","javascript",
                  "typescript","html","css","api","rest","graphql","docker","deployment"],
    "networks": ["network","tcp","udp","ip","http","https","dns","socket","routing",
                 "firewall","osi","packet","ssl","tls","websocket"],
    "dbms": ["dbms","database","rdbms","nosql","transaction","acid","normalization",
             "index","query","schema","table","relation","mongodb","postgres"],
    "sql": ["sql","select","join","where","group by","order by","insert","update",
            "delete","view","trigger","procedure","postgres","mysql"],
    "sqlite": ["sqlite","sqlite3","db","local database","mobile database","file database"],
    "os": ["os","operating system","process","thread","memory","virtual memory","paging",
           "scheduler","deadlock","mutex","semaphore","file system","kernel","linux"],
    "java": ["java","jvm","jdk","jre","spring","springboot","maven","gradle","class",
             "interface","abstract","garbage collection","thread","exception","collections"]
}


def tokenize(text: str) -> list[str]:
    """Lowercase, remove non-alphanum, split into tokens, remove stop words."""
    tokens = re.sub(r"[^a-z0-9#+.\-_]", " ", text.lower()).split()
    return [t for t in tokens if len(t) >= 2 and t not in STOP_WORDS]


def detect_topic(text: str) -> Optional[str]:
    """Detect the most likely topic from text content."""
    tl = text.lower()
    scores: dict[str, int] = {}
    for topic, kws in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in kws if re.search(r'\b' + re.escape(kw) + r'\b', tl))
        if score > 0:
            scores[topic] = score
    if not scores:
        return None
    return max(scores, key=lambda k: scores[k])


class BM25Retriever:
    """
    BM25 (Best Match 25) retrieval over chunked knowledge base documents.
    Provides fast keyword-based retrieval for context injection into LLM prompts.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self.chunks: list[Chunk] = []
        self.idf: dict[str, float] = {}
        self.avg_len: float = 0.0
        self._token_cache: list[list[str]] = []

    def load_knowledge_base(self, knowledge_dir: str,
                             chunk_size: int = 400,
                             overlap: int = 80):
        """
        Load all .txt and .md files from knowledge_dir, split into overlapping
        chunks, and build BM25 index.
        """
        self.chunks = []
        files = sorted([f for f in os.listdir(knowledge_dir)
                        if f.endswith(('.txt', '.md'))])

        if not files:
            print(f"[Retriever] WARNING: No knowledge files found in {knowledge_dir}")
            return

        for fname in files:
            topic = fname.replace('.txt', '').replace('.md', '').lower()
            fpath = os.path.join(knowledge_dir, fname)
            with open(fpath, encoding='utf-8') as f:
                text = f.read()

            # Split on markdown headers or Q: / A: markers to preserve structure
            sections = re.split(r'(?m)^#{1,4}\s+|^Q:\s*|^---+\s*$', text)

            for s_idx, section in enumerate(sections):
                section = section.strip()
                if len(section) < 50:  # skip tiny sections
                    continue

                lines = section.split('\n')
                title_line = lines[0].strip()[:80] if lines else f"{topic}_{s_idx}"

                # Sliding window chunks within section
                words = section.split()
                step = max(chunk_size - overlap, 1)
                for j in range(0, max(1, len(words)), step):
                    chunk_words = words[j: j + chunk_size]
                    if len(chunk_words) < 30:
                        continue
                    cid = f"{topic}_{s_idx}_{j}"
                    self.chunks.append(Chunk(
                        id=cid,
                        topic=topic,
                        title=title_line,
                        text=' '.join(chunk_words),
                    ))

        print(f"[Retriever] Loaded {len(self.chunks):,} chunks from {len(files)} files")
        self._build_index()

    def _build_index(self):
        """Build BM25 IDF scores for all terms."""
        self._token_cache = [tokenize(c.text) for c in self.chunks]
        doc_freq: dict[str, int] = defaultdict(int)
        doc_lens = [len(t) for t in self._token_cache]
        self.avg_len = sum(doc_lens) / max(len(doc_lens), 1)

        for toks in self._token_cache:
            for t in set(toks):  # count each term once per doc
                doc_freq[t] += 1

        N = len(self.chunks)
        self.idf = {
            t: math.log((N - df + 0.5) / (df + 0.5) + 1)
            for t, df in doc_freq.items()
        }
        print(f"[Retriever] BM25 index built: {len(self.idf):,} unique terms, "
              f"avg chunk length: {self.avg_len:.0f} tokens")

    def _bm25_score(self, query_terms: list[str], doc_idx: int) -> float:
        """Compute BM25 score for one document."""
        toks = self._token_cache[doc_idx]
        dl   = len(toks)
        tf_map: dict[str, int] = defaultdict(int)
        for t in toks:
            tf_map[t] += 1

        score = 0.0
        for term in query_terms:
            tf = tf_map[term]
            if tf == 0:
                continue
            norm_tf = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * dl / max(self.avg_len, 1))
            )
            score += self.idf.get(term, 0.0) * norm_tf
        return score

    def retrieve(self, query: str,
                 top_k: int = 5,
                 topic_filter: Optional[str] = None) -> list[Chunk]:
        """
        Retrieve top-k most relevant chunks for a query.
        Optionally filter to a specific topic.
        """
        if not self.chunks:
            return []

        query_terms = tokenize(query)
        if not query_terms:
            # No meaningful terms — return recent chunks from filtered topic
            pool = [c for c in self.chunks if topic_filter is None or c.topic == topic_filter]
            return pool[:top_k]

        scored: list[tuple[Chunk, float]] = []
        for i, chunk in enumerate(self.chunks):
            if topic_filter and chunk.topic != topic_filter:
                continue
            score = self._bm25_score(query_terms, i)
            if score > 0:
                scored.append((chunk, score))

        scored.sort(key=lambda x: -x[1])

        results: list[Chunk] = []
        for chunk, score in scored[:top_k]:
            results.append(Chunk(
                id=chunk.id, topic=chunk.topic,
                title=chunk.title, text=chunk.text,
                score=score,
            ))
        return results

    def format_context(self, chunks: list[Chunk], max_chars: int = 3500) -> str:
        """
        Format retrieved chunks into a context string for prompt injection.
        Deduplicates overlapping chunks.
        """
        parts: list[str] = []
        total = 0
        seen_titles: set[str] = set()

        for c in chunks:
            if c.title in seen_titles:
                continue
            seen_titles.add(c.title)
            section = f"[{c.topic.upper()} — {c.title}]\n{c.text}"
            if total + len(section) > max_chars:
                # Truncate last chunk to fit
                remaining = max_chars - total
                if remaining > 100:
                    parts.append(section[:remaining] + "...")
                break
            parts.append(section)
            total += len(section)

        return "\n\n---\n\n".join(parts)
