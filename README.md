# ChromaDB Playground

A minimal playground for exploring [ChromaDB](https://www.trychroma.com/) — an open-source vector database — with semantic similarity search using sentence embeddings.

## What it does

`grocery.py` demonstrates the core ChromaDB workflow:

1. **Creates an in-memory collection** with cosine similarity distance metric
2. **Embeds and stores** 14 grocery item descriptions using the `all-MiniLM-L6-v2` sentence transformer model
3. **Runs a similarity search** querying for items semantically similar to `["apple", "fresh"]`, returning the top 3 matches with their distances

## Project structure

```
.
├── grocery.py          # Main script
└── requirements.txt    # Python dependencies
```

## Setup

```bash
pip install -r requirements.txt
python grocery.py
```

> PyTorch is pulled from the CPU-only index to keep the install lightweight.

## Dependencies

| Package              | Version       |
|----------------------|---------------|
| chromadb             | 1.0.12        |
| sentence-transformers| 4.1.0         |
| torch                | latest (CPU)  |

## Example output

```
Creating collection...
Collection created: my_grocery_collection
Data added to collection.
Collection contents:
Number of items: 14
Collection 'my_grocery_collection' is ready for use.
Results for '['apple', 'fresh']':
...
Item ID: food_1, Document: fresh red apples, Distance: 0.1234
Item ID: food_13, Document: golden apple, Distance: 0.2345
Item ID: food_14, Document: red fruit, Distance: 0.3456
```