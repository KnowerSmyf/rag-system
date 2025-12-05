from config import settings # Import the real settings
from pathlib import Path

TEST_ROOT = Path(__file__).parent

# --- Overrides for Testing ---
CORPUS_FILE = TEST_ROOT / "fixtures" / "toy_corpus.jsonl"
STORES_DIR = TEST_ROOT / "output" / "stores"
SYNTHETIC_DATASET_DIR = TEST_ROOT / "output" / "synthetic_dataset"
MANUAL_GROUPS_FILE = TEST_ROOT / "fixtures" / "manual_document_groups_small.json"
RESULTS_DIR = TEST_ROOT / "output" / "results" # Redirect results too
LOG_DIR = TEST_ROOT / "output" / "logs"       # Redirect logs

AGENTIC_DB_PATH = STORES_DIR / "chroma_agentic_db"
TEXTSPLIT_DB_PATH = STORES_DIR / "chroma_text_split_db"
METADATA_DB_PATH = STORES_DIR / "chroma_metadata_db"
BM25_CACHE_FILE = STORES_DIR / "bm25_retriever.pkl"
CHUNK_CACHE_FILE = STORES_DIR / "agentic_chunks_cache.json"
COMPILED_AGENT_PATH = RESULTS_DIR / "optimized_programs" / "compiled_agentic_rag_test.json"
SYNTHETIC_TRAIN_SET_PATH = SYNTHETIC_DATASET_DIR / "train.jsonl"
SYNTHETIC_TEST_SET_PATH = SYNTHETIC_DATASET_DIR / "test.jsonl"

# Override generation numbers for speed
NUM_SINGLE_HOP = 2
TEST_SET_SIZE = 1

# --- Inherit Non-Path Settings ---
# Keep all other settings (model names, collection names, etc.) from the main config
# This uses a loop to copy all uppercase variables that haven't been overridden
_this_module = globals()
for key, value in settings.__dict__.items():
    if key.isupper() and key not in _this_module:
        _this_module[key] = value

# Add the as_dict() helper for compatibility
def as_dict():
    """
    Gathers all uppercase configuration variables from this module into a dictionary.
    """
    config_dict = {}
    # globals() is a built-in function that returns a dictionary of the current global symbol table
    for key, value in globals().items():
        # This is a convention: all config variables are uppercase
        if key.isupper():
            config_dict[key] = value
    return config_dict

