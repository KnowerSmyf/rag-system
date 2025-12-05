import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Tuple, Optional

# Import project modules
from src.data_processing.utils import load_corpus
from src.utils import get_active_config
from src.data_processing import cache_manager, indexing

# For agentic chunking and vector store
from langchain_google_vertexai import VertexAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hot-patch for sqlite3 on specific environments
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    logger.info("Successfully hot-patched sqlite3.") # Use logger
except ImportError:
    logger.warning("pysqlite3-binary not found, using system sqlite3.") # Use logger

# --- Agentic Chunking Prompt ---
AGENT_CHUNK_PROMPT = ChatPromptTemplate.from_template(
"""
You are an expert data scientist creating a knowledge base for a Retrieval-Augmented Generation (RAG) system. Your task is to analyze and chunk the following UQ policy document.

**Chunking Principles:**
1.  **Prioritize Fidelity:** Your primary goal is to create chunks using the **exact wording** from the source document. Do not summarize or rephrase.
2.  **Coherence & Completeness:** Each chunk must be thematically coherent and self-contained. Do not split lists, tables, or paragraphs mid-sentence.
3.  **Size:** Aim for chunks between 200 and 1500 characters.
4.  **CRITICAL RULE:** You MUST split the document into multiple chunks UNLESS the entire document is very short (e.g., less than 1000 characters).

**Output Format:**
Provide only the chunked document. Use "--- CHUNK ---" as a separator between each chunk.

**[SOURCE DOCUMENT]**
---
{source_document}
---

**[CHUNKED DOCUMENT]**
"""
)

# --- Helper Functions ---
def perform_chunking(
    body: str,
    metadata: dict,
    powerful_chunk_model_name: str
) -> Tuple[Optional[List[Document]], Optional[str]]: 
    """Performs agentic chunking using the specified powerful model."""
    try:
        chunking_llm = VertexAI(model_name=powerful_chunk_model_name, temperature=0.0)
        chunking_chain = AGENT_CHUNK_PROMPT | chunking_llm | StrOutputParser()
        full_response = chunking_chain.invoke({"source_document": body})
        individual_chunks = [c.strip() for c in full_response.split("--- CHUNK ---") if c.strip()]

        if not individual_chunks or (len(individual_chunks) == 1 and len(body) > 2000):
            # Return a special flag or None along with chunks to signal suboptimal result
            return None, f"Suboptimal chunking: Only {len(individual_chunks)} chunk(s) produced for long document."
            # if not individual_chunks: return [], "No chunks produced" # Alternative

        return [Document(page_content=content, metadata=metadata.copy()) for content in individual_chunks], None # Return chunks and no error

    except Exception as e:
        # Return None for chunks and the error message
        return None, f"Chunking failed: {e}"

# --- Main Workflow ---
def main(config_module=None):
    cfg = get_active_config(config_module)

    # --- STAGE 1: LOAD CACHE AND IDENTIFY NEW DOCUMENTS ---
    logger.info("[STAGE 1] Loading cache and identifying new documents")
    existing_cache, processed_titles = cache_manager.load_chunk_cache(cfg.CHUNK_CACHE_FILE)
    all_docs_from_corpus = load_corpus(cfg.CORPUS_FILE)
    docs_to_chunk = [doc for doc in all_docs_from_corpus if doc.get('title') not in processed_titles]
    
    # --- STAGE 2: PERFORM AGENTIC CHUNKING ON NEW DOCUMENTS ---
    logger.info("[STAGE 2] Performing Agentic Chunking")
    new_chunks = []
    if not docs_to_chunk:
        logger.info("✅ All documents are already chunked and cached. Moving on.")
    else:
        logger.info(f"Found {len(docs_to_chunk)} new documents to process.")
        chunking_errors = [] # List to store errors/warnings encountered during chunking
        suboptimal_warnings = []
        with ThreadPoolExecutor(max_workers=cfg.MAX_WORKERS) as executor:
            future_to_doc = {
                executor.submit(
                    perform_chunking,
                    doc['text'],
                    {"title": doc['title'], "url": doc['url']},
                    cfg.POWERFUL_CHUNK_MODEL
                ): doc for doc in docs_to_chunk
            }
            # Use tqdm without interleaving prints
            pbar = tqdm(as_completed(future_to_doc), total=len(docs_to_chunk), desc="Agentic Chunking (Batch)")
            for future in pbar:
                doc_info = future_to_doc[future]
                doc_title = doc_info.get('title', 'N/A')
                try:
                    # perform_chunking now returns (chunks, error_message_or_none)
                    result_chunks, error_msg = future.result()
                    
                    if result_chunks is not None:
                        new_chunks.extend(result_chunks)
                        if error_msg: # Log suboptimal warnings separately
                            suboptimal_warnings.append((doc_title, error_msg))
                    else: # An exception occurred or no chunks returned
                        chunking_errors.append((doc_title, error_msg or "Unknown chunking failure"))
                except Exception as exc: # Catch exceptions raised by future.result() itself
                    chunking_errors.append((doc_title, f"Future resolution exception: {exc}"))

        # --- Log errors/warnings AFTER the tqdm loop ---
        if suboptimal_warnings:
            logger.warning(f"{len(suboptimal_warnings)} documents had suboptimal chunking results:")
            for title, msg in suboptimal_warnings: # Iterate through the WHOLE list
                logger.error(f"  - '{title}': {msg}")             
        if chunking_errors:
            logger.error(f"{len(chunking_errors)} documents failed during chunking:")
            for title, msg in chunking_errors:
                logger.error(f"  - '{title}': {msg}")
        
        logger.info(f"Generated a total of {len(new_chunks)} new chunks from {len(docs_to_chunk)} attempted files.")

    # Combine existing and new chunks
    combined_cache = existing_cache + [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in new_chunks]
    if not combined_cache:
        logger.error("No chunks available. Cannot build stores. Exiting.")
        return
    
    # --- STAGE 3: UPDATE JSON CACHE (WRITE-AHEAD) ---
    logger.info("[STAGE 3] Updating/Saving the JSON chunk cache")
    if new_chunks:
        cache_manager.save_chunk_cache(cfg.CHUNK_CACHE_FILE, combined_cache)
    else:
        logger.info("No need to update cache. Moving on.")

    # --- STAGE 4: ALWAYS REBUILD VECTOR STORE ---
    logger.info(f"[STAGE 4] Rebuilding Chroma Vector Store: {cfg.AGENTIC_COLLECTION}")
    indexing.build_chroma_vectorstore(
        chunks=combined_cache,
        db_path=cfg.AGENTIC_DB_PATH,
        collection_name=cfg.AGENTIC_COLLECTION,
        embedding_model_name=cfg.EMBEDDING_MODEL,
        batch_size=cfg.EMBEDDING_BATCH_SIZE
    )

    # --- STAGE 5: REBUILD BM25 RETRIEVER FROM COMBINED CACHE ---
    logger.info("[STAGE 5] Rebuilding BM25 Retriever")
    indexing.build_bm25_index(
        chunks=combined_cache,
        save_path=cfg.AGENTIC_BM25_PATH
    )

    logger.info("All data stores are now synchronized (Vector Store and BM25 rebuilt).")

if __name__ == "__main__":
    main()