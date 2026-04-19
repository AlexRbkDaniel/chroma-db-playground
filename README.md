# ChromaDB Playground

A playground for exploring [ChromaDB](https://www.trychroma.com/) — an open-source vector database — with semantic similarity search, metadata filtering, and combined queries using sentence embeddings.

## Project structure

```
.
├── src/
│   ├── chroma/
│   │   └── collection_manager.py   # Reusable ChromaCollectionManager class
│   └── script/
│       ├── grocery.py              # Grocery similarity search
│       ├── employee.py             # Employee search with advanced queries
│       └── books.py                # Book search with advanced queries
├── resources/
│   └── data/
│       └── mock_data.py            # Mock datasets (grocery, employees, books)
├── .venv/                          # Virtual environment
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> PyTorch is pulled from the CPU-only index to keep the install lightweight.

## ChromaCollectionManager

All scripts share a single reusable class in `src/chroma/collection_manager.py`:

| Method | Description |
|--------|-------------|
| `create()` | Creates the ChromaDB collection |
| `add(documents, ids, metadatas)` | Adds documents with optional metadata |
| `get(where=None)` | Retrieves all documents, with optional metadata filter |
| `query(query_texts, n_results, where=None)` | Semantic similarity search with optional metadata filter |
| `update(ids, documents, metadatas)` | Updates existing documents |
| `delete(ids)` | Deletes documents by ID |

## Scripts

### grocery.py

Demonstrates the core ChromaDB workflow with 14 grocery item descriptions.

- Creates an in-memory collection with cosine similarity
- Embeds items using `all-MiniLM-L6-v2`
- Runs a similarity search for `["apple", "fresh"]`

```bash
python -m src.script.grocery
```

### employee.py

Demonstrates advanced search over 15 employee records.

- **Similarity search** — find employees by natural language (e.g. "Python developer with web experience")
- **Metadata filtering** — filter by department, experience range, or location
- **Combined search** — similarity query scoped by metadata filters (e.g. 8+ years experience in tech cities)

```bash
python -m src.script.employee
```

### books.py

Demonstrates advanced search over 8 books across Classic, Dystopian, Fantasy, and Science Fiction genres.

- **Similarity search** — find books by theme (e.g. "magical fantasy adventure")
- **Metadata filtering** — filter by genre, rating, publication decade, or page count
- **Combined search** — similarity query scoped by genre and rating filters

```bash
python -m src.script.books
```

## Dependencies

| Package               | Version      |
|-----------------------|--------------|
| chromadb              | 1.0.12       |
| sentence-transformers | 4.1.0        |
| torch                 | latest (CPU) |