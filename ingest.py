"""Document ingestion, chunking, and local embedding storage for local-study-rag."""

from __future__ import annotations

import json
import pathlib
import sqlite3
import urllib.request
from typing import Iterable


def extract_pages(file_path: pathlib.Path) -> list[tuple[int, str]]:
    """Return list of (page_number, text) tuples from .md, .txt, or .pdf files."""
    ext = file_path.suffix.lower()
    if ext in {".md", ".txt"}:
        return [(1, file_path.read_text(encoding="utf-8", errors="ignore"))]
    if ext == ".pdf":
        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install pypdf (`pip install pypdf`) to ingest PDF files.") from exc
        reader = PdfReader(str(file_path))
        return [(idx + 1, page.extract_text() or "") for idx, page in enumerate(reader.pages)]
    return []


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    step = max(chunk_size - overlap, 100)
    for start in range(0, len(words), step):
        piece = " ".join(words[start : start + chunk_size]).strip()
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(words):
            break
    return chunks


def fetch_embedding(text: str, base_url: str, model: str) -> list[float]:
    url = f"{base_url.rstrip('/')}/embeddings"
    body = json.dumps({"model": model, "input": text}).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return [float(x) for x in payload["data"][0]["embedding"]]


def init_db(db_path: pathlib.Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            page INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
        """
    )
    return conn


def ingest_directory(docs_dir: pathlib.Path, db_path: pathlib.Path, base_url: str, embed_model: str) -> int:
    files: Iterable[pathlib.Path] = sorted(
        p for p in docs_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".pdf", ".md", ".txt"}
    )
    conn = init_db(db_path)
    conn.execute("DELETE FROM chunks")
    count = 0
    for file_path in files:
        for page_num, text in extract_pages(file_path):
            for piece in chunk_text(text):
                vec = fetch_embedding(piece, base_url, embed_model)
                conn.execute(
                    "INSERT INTO chunks (source, page, content, embedding) VALUES (?, ?, ?, ?)",
                    (str(file_path.relative_to(docs_dir)), page_num, piece, json.dumps(vec)),
                )
                count += 1
    conn.commit()
    conn.close()
    return count
