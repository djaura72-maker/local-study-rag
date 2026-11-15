#!/usr/bin/env python3
"""100% offline CLI RAG study assistant powered by LM Studio or Ollama."""

from __future__ import annotations

import argparse
import pathlib
import sqlite3

from ingest import ingest_directory
from retriever import chat_complete, search_chunks

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_CHAT_MODEL = "qwen3.5-9b"


def cmd_ingest(args: argparse.Namespace) -> int:
    docs_dir = pathlib.Path(args.docs).expanduser().resolve()
    if not docs_dir.exists():
        print(f"Docs folder not found: {docs_dir}")
        return 1
    count = ingest_directory(docs_dir, pathlib.Path(args.db), args.base_url, args.embed_model)
    print(f"Indexed {count} chunks into {args.db}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    chunks = search_chunks(args.question, pathlib.Path(args.db), args.base_url, args.embed_model, top_k=args.top_k)
    if not chunks:
        print("No matching chunks found.")
        return 1

    context_block = "\n\n".join(
        f"[{c.source} | p.{c.page}]\n{c.content}" for c in chunks
    )
    prompt = (
        "Answer the study question using ONLY the context below. "
        "Cite sources inline as [filename | p.X].\n\n"
        f"Context:\n{context_block}\n\nQuestion: {args.question}"
    )
    answer = chat_complete(prompt, args.base_url, args.chat_model)
    print(answer)
    print("\nSources:")
    for c in chunks:
        print(f"- {c.source} (p.{c.page}, score={c.score:.3f})")
    return 0


def cmd_quiz(args: argparse.Namespace) -> int:
    db_path = pathlib.Path(args.db)
    if not db_path.exists():
        print(f"Index not found at {db_path}. Run `python3 rag.py ingest` first.")
        return 1

    conn = sqlite3.connect(str(db_path))
    rows = conn.execute("SELECT source, page, content FROM chunks ORDER BY RANDOM() LIMIT 5").fetchall()
    conn.close()
    if not rows:
        print("Index is empty.")
        return 1

    context_block = "\n\n".join(f"[{src} | p.{pg}]\n{txt}" for src, pg, txt in rows)
    prompt = (
        "Create 5 multiple-choice practice questions (A-D) based strictly on these study excerpts, "
        "followed by a concise Answer Key citing [filename | p.X].\n\n"
        f"{context_block}"
    )
    print(chat_complete(prompt, args.base_url, args.chat_model))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline RAG study & exam prep assistant.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Local OpenAI-compatible server URL.")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Embedding model ID.")
    parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL, help="Chat model ID.")
    parser.add_argument("--db", default="data/study_index.db", help="SQLite index path.")

    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Index PDF, Markdown, and text files from docs/.")
    p_ingest.add_argument("--docs", default="docs", help="Directory containing study files.")
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="Ask a question with cited answers.")
    p_ask.add_argument("question", help="Study question to answer.")
    p_ask.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve.")
    p_ask.set_defaults(func=cmd_ask)

    p_quiz = sub.add_parser("quiz", help="Generate 5 practice MCQs from indexed notes.")
    p_quiz.set_defaults(func=cmd_quiz)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
