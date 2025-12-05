import json
import shutil
import logging
from pathlib import Path
from typing import List, Dict

from src.utils import get_active_config
from src.data_processing import indexing

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def load_corpus_metadata(corpus_path: Path) -> List[Dict]:
    """
    Loads metadata (title, description, url) for all documents from a JSONL file.

    Args:
        corpus_path: Path to the scraped_corpus.jsonl file.

    Returns:
        A list of metadata dictionaries.

    Raises:
        FileNotFoundError: If the corpus file does not exist.
        json.JSONDecodeError: If a line in the file is not valid JSON.
        Exception: For other potential file reading errors.
    """
    logger.info(f"Loading metadata directly from corpus: {corpus_path}")
    metadata_list = []
    line_num = 0
    try:
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found at '{corpus_path}'")
             
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_num += 1
                doc = json.loads(line) # Can raise JSONDecodeError
                metadata_list.append({
                    "title": doc.get("title", f"Unknown Title Line {line_num}"), # Add fallback
                    "description": doc.get("description", ""),
                    "url": doc.get("url", "")
                })
        logger.info(f"✅ Loaded metadata for {len(metadata_list)} documents.")
        return metadata_list
    except FileNotFoundError as e:
        logger.error(e)
        raise # Re-raise for main function to handle
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON on line {line_num} in '{corpus_path}': {e}")
        raise # Re-raise
    except Exception as e:
        logger.error(f"Error reading corpus file '{corpus_path}': {e}", exc_info=True)
        raise # Re-raise unexpected errors

# --- Main Workflow ---
def main(config_module=None):
    """
    Builds the Chroma vector store containing embeddings of document metadata
    (title and description). Always rebuilds the store from scratch.
    """
    cfg = get_active_config(config_module)

    # Safety check before destructive operation
    if cfg.METADATA_DB_PATH.exists():
        # Keep confirmation prompt for safety when run manually
        overwrite = input(f"⚠️ Existing metadata database found at '{cfg.METADATA_DB_PATH}'. Overwrite and rebuild? (y/N): ").strip().lower()
        if overwrite == 'y':
            logger.warning("Removing existing metastore...")
            try:
                shutil.rmtree(cfg.METADATA_DB_PATH)
            except OSError as e:
                logger.error(f"Error removing directory '{cfg.METADATA_DB_PATH}': {e}", exc_info=True)
                return # Exit if removal fails
        else:
            logger.info("Rebuild aborted by user.")
            return # Exit if user says no

    # 1. Load metadata
    try:
        all_metadata = load_corpus_metadata(cfg.CORPUS_FILE)
    except Exception:
        logger.error("Failed to load corpus metadata. Cannot proceed.")
        return # Exit if loading fails

    if not all_metadata:
        logger.warning("No metadata loaded from corpus file. Nothing to index.")
        return

    # 2. Prepare chunks for indexing
    chunks_for_indexing = [
        {
            "page_content": f"Title: {meta['title']} Description: {meta.get('description', '') or 'No description available.'}",
            "metadata": meta
        }
        for meta in all_metadata
    ]

    # 3. Build Chroma Vector Store using the shared indexing function
    logger.info("Building Chroma Metadata Vector Store")
    try:
        indexing.build_chroma_vectorstore( # Make sure indexing is imported
            chunks=chunks_for_indexing,
            db_path=cfg.METADATA_DB_PATH,
            collection_name=cfg.METADATA_COLLECTION,
            embedding_model_name=cfg.EMBEDDING_MODEL,
            batch_size=cfg.EMBEDDING_BATCH_SIZE
        )
        logger.info(f"✅ Metadata vector store build complete.")
    except Exception as e:
        logger.error(f"Failed to build Chroma vector store: {e}", exc_info=True)

if __name__ == "__main__":
    main()