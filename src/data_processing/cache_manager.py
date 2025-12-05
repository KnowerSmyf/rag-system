import json
import logging
from pathlib import Path
from typing import List, Dict, Set, Tuple

logger = logging.getLogger(__name__)

def load_chunk_cache(cache_path: Path) -> Tuple[List[Dict], Set[str]]:
    """Loads the chunk cache file and returns the cache data and processed titles."""
    existing_cache = []
    processed_titles = set()
    if cache_path.exists():
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                existing_cache = json.load(f)
            processed_titles = {item['metadata']['title'] for item in existing_cache if 'metadata' in item and 'title' in item['metadata']}
            logger.info(f"Loaded {len(existing_cache)} existing chunks from {len(processed_titles)} documents in cache.")
        except (FileNotFoundError, json.JSONDecodeError):
            logger.warning(f"Cache file {cache_path} exists but could not be loaded. Treating as empty.")
            existing_cache = []
            processed_titles = set()
    else:
        logger.info("No existing chunk cache found.")
    return existing_cache, processed_titles

def save_chunk_cache(cache_path: Path, combined_cache: List[Dict]) -> None:
    """Saves the combined chunk data to the cache file."""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(combined_cache, f, indent=2)
        logger.info(f"JSON cache saved. Total chunks: {len(combined_cache)} to {cache_path}.")
    except Exception as e:
        logger.error(f"Failed to save chunk cache to {cache_path}: {e}", exc_info=True)