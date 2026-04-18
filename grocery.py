import chromadb
from chromadb.types import Collection
from chromadb.utils import embedding_functions

grocery_items = [
    'fresh red apples',
    'organic bananas',
    'ripe mangoes',
    'whole wheat bread',
    'farm-fresh eggs',
    'natural yogurt',
    'frozen vegetables',
    'grass-fed beef',
    'free-range chicken',
    'fresh salmon fillet',
    'aromatic coffee beans',
    'pure honey',
    'golden apple',
    'red fruit'
]

ids = [f"food_{index + 1}" for index, _ in enumerate(grocery_items)]

ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.Client()
collection_name = "my_grocery_collection"


def create_collection() -> Collection | None:
    try:
        print("Creating collection...")
        collection = client.create_collection(
            name=collection_name,
            embedding_function=ef,
            metadata={
                "description": "A collection for storing grocery data",
                "hnsw:space": "cosine"
            }
        )
        print(f"Collection created: {collection.name}")
        collection.add(
            documents=grocery_items,
            ids=ids,
            metadatas=[{"source": "grocery_store", "category": "food"} for _ in grocery_items]
        )
        print("Data added to collection.")
        all_items = collection.get()
        print("Collection contents:")
        print(f"Number of items: {len(all_items['documents'])}")
        return collection
    except Exception as error:
        print(f"Error in collection creation: {error}")
        return None


def perform_similarity_search(_collection: Collection, query_term: list[str]):
    try:
        results = _collection.query(
            query_texts=query_term,
            n_results=3
        )
        print(f"Results for '{query_term}':")
        print("Raw results:")
        print(results)
        print("Formatted results:")
        for item_id, document, distance in zip(results['ids'][0], results['documents'][0], results['distances'][0]):
            print(f"Item ID: {item_id}, Document: {document}, Distance: {distance:.4f}")
    except Exception as error:
        print(f"Error in similarity search: {error}")


collection: Collection | None = create_collection()
if collection:
    print(f"Collection '{collection.name}' is ready for use.")
else:
    print("Failed to create collection. Please check the error messages.")

perform_similarity_search(collection, ["apple", "fresh"])
