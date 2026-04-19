from src.chroma.collection_manager import ChromaCollectionManager
from resources.data.mock_data import grocery_items

manager = ChromaCollectionManager(
    collection_name="my_grocery_collection",
    metadata={
        "description": "A collection for storing grocery data",
        "hnsw:space": "cosine",
    },
)

manager.create()
manager.add(
    documents=grocery_items,
    ids=[f"food_{i + 1}" for i, _ in enumerate(grocery_items)],
    metadatas=[{"source": "grocery_store", "category": "food"} for _ in grocery_items]
)

items = manager.get()
print(f"Collection contains {len(items['documents'])} items.")

results = manager.query(query_texts=["apple", "fresh"])

if results:
    print("Similarity search results:")
    for item_id, document, distance in zip(
            results["ids"][0], results["documents"][0], results["distances"][0]
    ):
        print(f"  ID: {item_id}, Document: {document}, Distance: {distance:.4f}")