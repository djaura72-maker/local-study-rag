# local-study-rag

A 100% offline **Retrieval-Augmented Generation (RAG)** study and exam-prep CLI powered by **LM Studio** (`http://localhost:1234/v1`) or **Ollama** using `nomic-embed-text` and a local SQLite vector store.

## Features

- **`ingest`:** Extracts text from PDFs, Markdown, and `.txt` files in `docs/`, chunks by word count with overlap, and stores embeddings locally in SQLite.
- **`ask`:** Performs cosine-similarity search and answers questions with exact `[filename | p.X]` citations.
- **`quiz`:** Generates 5 multiple-choice practice questions and an answer key from indexed study material.
- **Minimal Dependencies:** Uses Python's standard library + SQLite (`pypdf` needed only for PDFs).

## Quick Start

```bash
# 1. Install PDF support (optional if using .md/.txt)
pip install -r requirements.txt

# 2. Place study notes or past papers in docs/
mkdir -p docs

# 3. Index your files (with LM Studio serving nomic-embed-text)
python3 rag.py ingest --docs docs

# 4. Ask a question or generate a practice quiz
python3 rag.py ask "Explain ASP.NET page life cycle events"
python3 rag.py quiz
```

## Repository Structure

```text
.
├── rag.py            # CLI entry point (ingest, ask, quiz)
├── ingest.py         # Document extraction, chunking, and SQLite storage
├── retriever.py      # Cosine similarity search and local chat completions
├── requirements.txt  # Optional pypdf dependency
├── README.md         # Documentation and usage guide
├── CHANGELOG.md      # Release history
├── CONTRIBUTING.md   # Contribution guidelines
├── LICENSE           # MIT License
└── .gitignore        # Python, data/, docs/, and macOS ignores
```

## License

MIT License — see [`LICENSE`](LICENSE).
