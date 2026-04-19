from resources.data.mock_data import books
from src.chroma.collection_manager import ChromaCollectionManager


def main():
    manager = ChromaCollectionManager(
        collection_name="book_collection",
        metadata={
            "description": "A collection for storing book data",
            "hnsw:space": "cosine",
        }
    )

    print(f"Initialized Chroma collection manager for '{manager.collection_name}'.")

    manager.create()

    book_documents = [
        f"{book['title']} by {book['author']}. {book['description']} "
        f"Themes: {book['themes']}. Setting: {book['setting']}."
        for book in books
    ]

    manager.add(
        ids=[book["id"] for book in books],
        documents=book_documents,
        metadatas=[{
            "title": book["title"],
            "author": book["author"],
            "genre": book["genre"],
            "year": book["year"],
            "rating": book["rating"],
            "pages": book["pages"],
            "themes": book["themes"],
            "setting": book["setting"],
        } for book in books]
    )

    all_items = manager.get()
    print(f"Collection contains {len(all_items['documents'])} books.\n")

    perform_advanced_search(manager)


def perform_advanced_search(manager: ChromaCollectionManager):
    try:
        print("=== Similarity Search Examples ===")

        print("\n1. Searching for magical fantasy adventure:")
        query_text = "magical fantasy adventure with wizards and epic quests"
        results = manager.query(query_texts=[query_text], n_results=3)
        print(f"Query: '{query_text}'")
        for i, (doc_id, document, distance) in enumerate(zip(
                results["ids"][0], results["documents"][0], results["distances"][0]
        )):
            metadata = results["metadatas"][0][i]
            print(f"  {i + 1}. {metadata['title']} by {metadata['author']} ({doc_id}) - Distance: {distance:.4f}")
            print(f"     Genre: {metadata['genre']}, Rating: {metadata['rating']}")
            print(f"     Document: {document[:100]}...")

        print("\n2. Searching for dark dystopian societies:")
        query_text = "dark dystopian society control oppression survival"
        results = manager.query(query_texts=[query_text], n_results=3)
        print(f"Query: '{query_text}'")
        for i, (doc_id, document, distance) in enumerate(zip(
                results["ids"][0], results["documents"][0], results["distances"][0]
        )):
            metadata = results["metadatas"][0][i]
            print(f"  {i + 1}. {metadata['title']} by {metadata['author']} ({doc_id}) - Distance: {distance:.4f}")
            print(f"     Genre: {metadata['genre']}, Year: {metadata['year']}")

        print("\n=== Metadata Filtering Examples ===")

        print("\n3. Finding Fantasy and Science Fiction books:")
        results = manager.get(where={"genre": {"$in": ["Fantasy", "Science Fiction"]}})
        print(f"Found {len(results['ids'])} Fantasy/Sci-Fi books:")
        for i, doc_id in enumerate(results["ids"]):
            metadata = results["metadatas"][i]
            print(f"  - {metadata['title']} ({metadata['genre']}, {metadata['year']})")

        print("\n4. Finding books rated 4.3 or higher:")
        results = manager.get(where={"rating": {"$gte": 4.3}})
        print(f"Found {len(results['ids'])} highly-rated books:")
        for i, doc_id in enumerate(results["ids"]):
            metadata = results["metadatas"][i]
            print(f"  - {metadata['title']} — Rating: {metadata['rating']}")

        print("\n5. Finding books published between 1950 and 2000:")
        results = manager.get(where={"$and": [{"year": {"$gte": 1950}}, {"year": {"$lte": 2000}}]})
        print(f"Found {len(results['ids'])} books from 1950–2000:")
        for i, doc_id in enumerate(results["ids"]):
            metadata = results["metadatas"][i]
            print(f"  - {metadata['title']} ({metadata['year']})")

        print("\n6. Finding shorter reads (under 300 pages):")
        results = manager.get(where={"pages": {"$lt": 300}})
        print(f"Found {len(results['ids'])} books under 300 pages:")
        for i, doc_id in enumerate(results["ids"]):
            metadata = results["metadatas"][i]
            print(f"  - {metadata['title']} — {metadata['pages']} pages")

        print("\n=== Combined Search: Similarity + Metadata Filtering ===")

        print("\n7. Finding highly-rated dystopian books about control and rebellion:")
        query_text = "society control rebellion freedom totalitarianism"
        results = manager.query(
            query_texts=[query_text],
            n_results=5,
            where={
                "$and": [
                    {"genre": "Dystopian"},
                    {"rating": {"$gte": 4.0}}
                ]
            }
        )
        print(f"Query: '{query_text}' with filters (Dystopian, rating ≥ 4.0)")
        print(f"Found {len(results['ids'][0])} matching books:")
        for i, (doc_id, document, distance) in enumerate(zip(
                results["ids"][0], results["documents"][0], results["distances"][0]
        )):
            metadata = results["metadatas"][0][i]
            print(f"  {i + 1}. {metadata['title']} by {metadata['author']} ({doc_id}) - Distance: {distance:.4f}")
            print(f"     Rating: {metadata['rating']}, Year: {metadata['year']}")
            print(f"     Document snippet: {document[:80]}...")

        print("\n8. Finding classic literature about social themes (bonus):")
        query_text = "social injustice class inequality moral growth"
        results = manager.query(
            query_texts=[query_text],
            n_results=3,
            where={
                "$and": [
                    {"genre": "Classic"},
                    {"rating": {"$gte": 4.0}}
                ]
            }
        )
        print(f"Query: '{query_text}' with filters (Classic, rating ≥ 4.0)")
        print(f"Found {len(results['ids'][0])} matching books:")
        for i, (doc_id, document, distance) in enumerate(zip(
                results["ids"][0], results["documents"][0], results["distances"][0]
        )):
            metadata = results["metadatas"][0][i]
            print(f"  {i + 1}. {metadata['title']} by {metadata['author']} - Distance: {distance:.4f}")
            print(f"     Themes: {metadata['themes']}")

    except Exception as error:
        print(f"Error in advanced search: {error}")


if __name__ == "__main__":
    main()
