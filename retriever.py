"""Vector similarity retrieval and local LLM completion helpers."""

from __future__ import annotations

import json
import math
import pathlib
import sqlite3
import urllib.request
from dataclasses import dataclass

from ingest import fetch_embedding


@dataclass
class RetrievedChunk:
    source: str
    page: int
    content: str
    score: float


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def search_chunks(
    query: str,
    db_path: pathlib.Path,
    base_url: str,
    embed_model: str,
    top_k: int = 4,
) -> list[RetrievedChunk]:
    if not db_path.exists():
        raise FileNotFoundError(f"Index not found at {db_path}. Run `python3 rag.py ingest` first.")

    q_vec = fetch_embedding(query, base_url, embed_model)
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute("SELECT source, page, content, embedding FROM chunks").fetchall()
    conn.close()

    scored: list[RetrievedChunk] = []
    for source, page, content, emb_json in rows:
        score = cosine_similarity(q_vec, json.loads(emb_json))
        scored.append(RetrievedChunk(source=source, page=page, content=content, score=score))

    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[:top_k]


def chat_complete(prompt: str, base_url: str, chat_model: str) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    body = json.dumps(
        {
            "model": chat_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 700,
        }
    ).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return str(payload["choices"][0]["message"]["content"]).strip()
