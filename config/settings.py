from pathlib import Path

# --- Core Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
SRC_DIR = PROJECT_ROOT / "src"
LOG_DIR = PROJECT_ROOT / "logs"

# --- Data Subdirectories & Files ---
CORPUS_FILE = DATA_DIR / "scraped_corpus.jsonl"
STORES_DIR = DATA_DIR / "stores"

# --- Data Store Paths ---
AGENTIC_DB_PATH = STORES_DIR / "chroma_agentic_db"
TEXTSPLIT_DB_PATH = STORES_DIR / "chroma_text_split_db"
METADATA_DB_PATH = STORES_DIR / "chroma_metadata_db"
AGENTIC_BM25_PATH = STORES_DIR / "bm25_agentic_retriever.pkl"
TEXTSPLIT_BM25_PATH = STORES_DIR / "bm25_textsplit_retriever.pkl"
CHUNK_CACHE_FILE = STORES_DIR / "agentic_chunks_cache.json"

# --- Model & Training Settings ---
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
FAST_CHUNK_MODEL = "gemini-2.5-flash"
POWERFUL_CHUNK_MODEL = "gemini-2.5-pro"

# --- Collection Names ---
AGENTIC_COLLECTION = "PPL_agentic_documents"
TEXTSPLIT_COLLECTION = "PPL_text_split_documents"
METADATA_COLLECTION = "PPL_document_metadata"

# --- Process Settings ---
EMBEDDING_BATCH_SIZE = 128
MAX_WORKERS = 6
TEXTSPLIT_CHUNK_SIZE = 1000
TEXTSPLIT_CHUNK_OVERLAP = 200

# --- QA Dataset Generation Settings ---
TEACHER_LM_MODEL="anthropic/claude-sonnet-4-20250514"
TEACHER_LM_KWARGS = {"temperature": 0.7, "max_tokens": 4000}
CLASSIFIER_LM_MODEL="anthropic/claude-3-haiku-20240307"
CLASSIFIER_LM_KWARGS= {"max_tokens": 256, "temperature": 0.0}

SYNTHETIC_DATASET_DIR = DATA_DIR / "synthetic_dataset"
MANUAL_GROUPS_FILE = SYNTHETIC_DATASET_DIR / "manual_document_groups.json"
NUM_SINGLE_HOP = 100
TEST_SET_SIZE = 50

# --- Training & Evaluation Settings ---
COMPILER_LM_MODEL = "gemini/gemini-2.5-pro"   # A powerful model is best for compilation
RAG_LM_MODEL = "gemini/gemini-2.5-flash"      # A faster model for the main agent
EVALUATOR_LM_MODEL = "openai/gpt-5-mini"
EVALUATOR_LM_KWARGS = {
    "model": "openai/gpt-5-mini",
    "temperature": 1.0,
    "max_tokens": 16_000  # Satisfies the requirement
}

# SYNTHETIC_SET_PATH = DATA_DIR / "synthetic_dataset"
SYNTHETIC_SET_PATH = PROJECT_ROOT / "multi_hops"
SYNTHETIC_TRAIN_SET_PATH = SYNTHETIC_SET_PATH / "train.jsonl"
SYNTHETIC_TEST_SET_PATH = SYNTHETIC_SET_PATH / "test.jsonl"
COMPILED_AGENT_PATH = RESULTS_DIR / "optimized_programs" / "compiled_retriever_bootstrap_v4.json"

# --- Scraping stuff ---
# TODO: Consider adding the safety settings here as Kwargs too?
# SCRAPING_HELPER_MODEL = "gemini/gemini-2.5-flash" <- with the kwargs, this might be pointless
PPL_BROWSE_URL = "https://policies.uq.edu.au/browse"
SCRAPING_THROTTLE_SECONDS = 1 # polite delay between requests
SCRAPING_HELPER_MODEL_KWARGS = {
    "model": "gemini/gemini-2.5-flash",
    "max_tokens": 16_000,
    "safety_settings": [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
}

# --- Helper Functions ---
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

