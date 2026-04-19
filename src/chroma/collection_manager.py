import chromadb
from chromadb.types import Collection
from chromadb.utils import embedding_functions


class ChromaCollectionManager:
    def __init__(
            self,
            collection_name: str,
            model_name: str = "all-MiniLM-L6-v2",
            metadata: dict | None = None,
    ):
        self._client = chromadb.Client()
        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
        self._collection_name = collection_name
        self._metadata = metadata or {"hnsw:space": "cosine"}
        self._collection: Collection | None = None

    def create(self) -> Collection | None:
        try:
            self._collection = self._client.create_collection(
                name=self._collection_name,
                embedding_function=self._ef,
                metadata=self._metadata,
            )
            print(f"Collection '{self._collection.name}' created.")
            return self._collection
        except Exception as error:
            print(f"Error creating collection: {error}")
            return None

    def add(self, documents: list[str], ids: list[str], metadatas: list[dict] | None = None) -> None:
        if not self._collection:
            print("No active collection. Call create() first.")
            return
        try:
            self._collection.add(documents=documents, ids=ids, metadatas=metadatas)
            print(f"Added {len(documents)} documents to '{self._collection_name}'.")
        except Exception as error:
            print(f"Error adding documents: {error}")

    def get(self, where: dict | None = None) -> dict | None:
        if not self._collection:
            print("No active collection.")
            return None
        try:
            kwargs = {}
            if where:
                kwargs["where"] = where
            return self._collection.get(**kwargs)
        except Exception as error:
            print(f"Error fetching collection: {error}")
            return None

    def query(self, query_texts: list[str], n_results: int = 3, where: dict | None = None) -> dict | None:
        if not self._collection:
            print("No active collection.")
            return None
        try:
            kwargs: dict = {"query_texts": query_texts, "n_results": n_results}
            if where:
                kwargs["where"] = where
            return self._collection.query(**kwargs)
        except Exception as error:
            print(f"Error querying collection: {error}")
            return None

    def update(self, ids: list[str], documents: list[str], metadatas: list[dict] | None = None) -> None:
        if not self._collection:
            print("No active collection.")
            return
        try:
            self._collection.update(ids=ids, documents=documents, metadatas=metadatas)
            print(f"Updated {len(ids)} documents in '{self._collection_name}'.")
        except Exception as error:
            print(f"Error updating documents: {error}")

    def delete(self, ids: list[str]) -> None:
        if not self._collection:
            print("No active collection.")
            return
        try:
            self._collection.delete(ids=ids)
            print(f"Deleted {len(ids)} documents from '{self._collection_name}'.")
        except Exception as error:
            print(f"Error deleting documents: {error}")

    @property
    def collection_name(self):
        return self._collection_name
