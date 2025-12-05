import logging
import sys

# Import project modules
from src.data_processing.utils import load_corpus
from src.data_processing import indexing # Import our new indexing functions
from src.utils import get_active_config

# Import LangChain components
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Hot-patch for sqlite3 ---
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    logger.info("Successfully hot-patched sqlite3.")
except ImportError:
    logger.warning("pysqlite3-binary not found, using system sqlite3.")

# --- Main Workflow ---
def main(config_module=None):
    """
    Builds the text-split vector store and BM25 index from the full corpus.
    This script always rebuilds the stores from scratch to ensure consistency.
    """
    cfg = get_active_config(config_module)

    # --- STAGE 1: LOAD AND SPLIT DOCUMENTS ---
    logger.info("[STAGE 1] Loading and splitting all documents from corpus")
    try:
        all_docs_from_corpus = load_corpus(cfg.CORPUS_FILE)
    except FileNotFoundError:
        logger.error(f"Corpus file not found at {cfg.CORPUS_FILE}. Exiting.")
        return

    # Convert raw dicts to LangChain Document objects for splitting
    lc_docs = [Document(page_content=doc['text'], metadata=doc) for doc in all_docs_from_corpus]

    # Initialize the text splitter using settings from config
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.TEXTSPLIT_CHUNK_SIZE,
        chunk_overlap=cfg.TEXTSPLIT_CHUNK_OVERLAP,
        add_start_index=True
    )
    all_chunks = text_splitter.split_documents(lc_docs)
    logger.info(f"Split {len(all_docs_from_corpus)} documents into {len(all_chunks)} chunks.")

    if not all_chunks:
        logger.error("No chunks were generated. Cannot build stores. Exiting.")
        return

    # Convert LangChain Document objects back to simple dicts for our indexing functions
    chunk_dicts = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in all_chunks]

    # --- STAGE 2: REBUILD VECTOR STORE ---
    logger.info("[STAGE 2] Rebuilding Chroma Vector Store")
    indexing.build_chroma_vectorstore(
        chunks=chunk_dicts,
        db_path=cfg.TEXTSPLIT_DB_PATH,
        collection_name=cfg.TEXTSPLIT_COLLECTION,
        embedding_model_name=cfg.EMBEDDING_MODEL,
        batch_size=cfg.EMBEDDING_BATCH_SIZE
    )

    # --- STAGE 3: REBUILD BM25 INDEX ---
    logger.info("[STAGE 3] Rebuilding BM25 Index")
    indexing.build_bm25_index(
        chunks=chunk_dicts,
        save_path=cfg.TEXTSPLIT_BM25_FILE
    )

    logger.info("Text-split data stores are now synchronized (Vector Store and BM25 rebuilt).")

if __name__ == "__main__":
    main()