from resources.data.mock_data import employees
from src.chroma.collection_manager import ChromaCollectionManager


def main():
    manager = ChromaCollectionManager(
        collection_name="employee_collection",
        metadata={
            "description": "A collection for storing employee data",
            "hnsw:space": "cosine",
        }
    )

    print(f"Initialized Chroma collection manager for '{manager.collection_name}'.")

    manager.create()

    employee_documents = []
    for employee in employees:
        document = f"{employee['role']} with {employee['experience']} years of experience in {employee['department']}. "
        document += f"Skills: {employee['skills']}. Located in {employee['location']}. "
        document += f"Employment type: {employee['employment_type']}."
        employee_documents.append(document)

    manager.add(
        ids=[employee["id"] for employee in employees],
        documents=employee_documents,
        metadatas=[{
            "name": employee["name"],
            "department": employee["department"],
            "role": employee["role"],
            "experience": employee["experience"],
            "location": employee["location"],
            "employment_type": employee["employment_type"]
        } for employee in employees]
    )

    all_items = manager.get()
    print(f"Collection contains {len(all_items['documents'])} employees.\n")

    perform_advanced_search(manager)


def perform_advanced_search(manager: ChromaCollectionManager):
    try:
        print("=== Similarity Search Examples ===")

        print("\n1. Searching for Python developers:")
        query_text = "Python developer with web development experience"
        results = manager.query(query_texts=[query_text], n_results=3)
        print(f"Query: '{query_text}'")
        for i, (doc_id, document, distance) in enumerate(zip(
                results["ids"][0], results["documents"][0], results["distances"][0]
        )):
            metadata = results["metadatas"][0][i]
            print(f"  {i + 1}. {metadata['name']} ({doc_id}) - Distance: {distance:.4f}")
            print(f"     Role: {metadata['role']}, Department: {metadata['department']}")
            print(f"     Document: {document[:100]}...")

        print("\n2. Searching for leadership and management roles:")
        query_text = "team leader manager with experience"
        results = manager.query(query_texts=[query_text], n_results=3)
        print(f"Query: '{query_text}'")
        for i, (doc_id, document, distance) in enumerate(zip(
                results["ids"][0], results["documents"][0], results["distances"][0]
        )):
            metadata = results["metadatas"][0][i]
            print(f"  {i + 1}. {metadata['name']} ({doc_id}) - Distance: {distance:.4f}")
            print(f"     Role: {metadata['role']}, Experience: {metadata['experience']} years")

        print("\n=== Metadata Filtering Examples ===")

        print("\n3. Finding all Engineering employees:")
        results = manager.get(where={"department": "Engineering"})
        print(f"Found {len(results['ids'])} Engineering employees:")
        for i, doc_id in enumerate(results["ids"]):
            metadata = results["metadatas"][i]
            print(f"  - {metadata['name']}: {metadata['role']} ({metadata['experience']} years)")

        print("\n4. Finding employees with 10+ years experience:")
        results = manager.get(where={"experience": {"$gte": 10}})
        print(f"Found {len(results['ids'])} senior employees:")
        for i, doc_id in enumerate(results["ids"]):
            metadata = results["metadatas"][i]
            print(f"  - {metadata['name']}: {metadata['role']} ({metadata['experience']} years)")

        print("\n5. Finding employees in California:")
        results = manager.get(where={"location": {"$in": ["San Francisco", "Los Angeles"]}})
        print(f"Found {len(results['ids'])} employees in California:")
        for i, doc_id in enumerate(results["ids"]):
            metadata = results["metadatas"][i]
            print(f"  - {metadata['name']}: {metadata['location']}")

        print("\n=== Combined Search: Similarity + Metadata Filtering ===")

        print("\n6. Finding senior Python developers in major tech cities:")
        query_text = "senior Python developer full-stack"
        results = manager.query(
            query_texts=[query_text],
            n_results=5,
            where={
                "$and": [
                    {"experience": {"$gte": 8}},
                    {"location": {"$in": ["San Francisco", "New York", "Seattle"]}}
                ]
            }
        )
        print(f"Query: '{query_text}' with filters (8+ years, major tech cities)")
        print(f"Found {len(results['ids'][0])} matching employees:")
        for i, (doc_id, document, distance) in enumerate(zip(
                results["ids"][0], results["documents"][0], results["distances"][0]
        )):
            metadata = results["metadatas"][0][i]
            print(f"  {i + 1}. {metadata['name']} ({doc_id}) - Distance: {distance:.4f}")
            print(f"     {metadata['role']} in {metadata['location']} ({metadata['experience']} years)")
            print(f"     Document snippet: {document[:80]}...")

    except Exception as error:
        print(f"Error in advanced search: {error}")


if __name__ == "__main__":
    main()
