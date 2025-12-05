import pickle
from typing import Dict
from pathlib import Path
import chromadb

# LangChain components for data interaction
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class PPLDataStores:
    """A class to handle the loading and initialization of all data stores."""
    def __init__(self, config: Dict):
        self.config = config
        print("--- Initializing PPL Data Stores ---")
        try:
            self.agentic_vectorstore = self._load_vectorstore(config['AGENTIC_DB_PATH'], config['AGENTIC_COLLECTION'])
            self.textsplit_vectorstore = self._load_vectorstore(config['TEXTSPLIT_DB_PATH'], config['TEXTSPLIT_COLLECTION'])
            self.metadata_vectorstore = self._load_vectorstore(config['METADATA_DB_PATH'], config['METADATA_COLLECTION'])
            self.agentic_bm25_retriever = self._load_bm25_retriever(config['AGENTIC_BM25_PATH'])
            self.textsplit_bm25_retriever = self._load_bm25_retriever(config['TEXTSPLIT_BM25_PATH'])
            print("✅ All data stores loaded successfully.")

        except (FileNotFoundError, ValueError) as e:
            print(f"❌ ERROR: Failed to load data stores. {e}")
            raise

    def _load_vectorstore(self, persist_dir: Path, collection_name: str) -> Chroma:
        # 1. Check if the persistence directory exists
        if not persist_dir.exists():
            raise FileNotFoundError(f"Chroma DB directory not found at: {persist_dir.resolve()}")

        # 2. Check if the collection exists within the directory
        client = chromadb.PersistentClient(path=persist_dir)
        collections = [c.name for c in client.list_collections()]
        if collection_name not in collections:
            raise ValueError(
                f"Collection '{collection_name}' not found in Chroma DB at {persist_dir}. "
                f"Available collections: {collections}"
            )

        print(f"  -> Loading collection '{collection_name}' from {persist_dir}...")
        embedding_function = HuggingFaceEmbeddings(model_name=self.config['EMBEDDING_MODEL'])
        return Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embedding_function,
            collection_name=collection_name
        )

    def _load_bm25_retriever(self, path: Path):
        """Loads a pickled BM25 retriever, raising an error if the file doesn't exist."""
        if not path.exists():
            raise FileNotFoundError(f"BM25 retriever file not found at: {path.resolve()}")

        print(f"  -> Loading BM25 retriever from {path}...")
        with open(path, 'rb') as f:
            return pickle.load(f)