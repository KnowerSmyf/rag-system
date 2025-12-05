import logging
import pickle
import shutil
from pathlib import Path
from typing import List, Dict
import spacy
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings

from .utils import cuda_device

logger = logging.getLogger(__name__)

def build_chroma_vectorstore(
    chunks: List[Dict], # Expects list of dicts from cache format
    db_path: Path,
    collection_name: str,
    embedding_model_name: str,
    batch_size: int = 128
) -> None:
    """Builds (or rebuilds) a Chroma vector store from chunks with pre-computed embeddings."""
    device = cuda_device()

    texts = [c.get('page_content', '') for c in chunks]
    metadatas = [c.get('metadata', {}) for c in chunks]

    # --- Rebuild Vector Store using from_texts ---
    if db_path.exists():
        logger.warning(f"Removing existing vector store at {db_path}")
        shutil.rmtree(db_path)

    if texts:
        logger.info(f"Creating new vector store at {db_path} using Chroma's internal embedding...")

        # Pass the wrapper with the desired device hint
        lc_embed_wrapper = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": device}, 
            encode_kwargs={
                "batch_size": batch_size, 
                "normalize_embeddings": True, 
            },
            show_progress=True
        )
        try:
            # Let Chroma.from_texts calculate embeddings using the wrapper
            Chroma.from_texts(
                texts=texts,
                embedding=lc_embed_wrapper, # Provide the function
                metadatas=metadatas,
                collection_name=collection_name,
                persist_directory=str(db_path)
            )
            logger.info(f"Successfully created vector store with {len(texts)} chunks.")
        except Exception as e:
            logger.error(f"Chroma.from_texts failed during embedding/creation: {e}", exc_info=True)
             
    else:
        logger.warning("No text chunks available to create vector store.")

def build_bm25_index(
    chunks: List[Dict], # Expects list of dicts from cache format
    save_path: Path
) -> None:
    """Builds and saves a BM25 retriever index from chunks."""
    if not chunks:
        logger.warning("No chunks available to build BM25 index. Skipping.")
        return

    logger.info("Initializing spaCy for lemmatization...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logger.error("spaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
        return

    all_docs_for_bm25 = [Document(page_content=c.get('page_content',''), metadata=c.get('metadata',{})) for c in chunks]
    lemmatized_docs = []
    for spacy_doc in tqdm(nlp.pipe([doc.page_content for doc in all_docs_for_bm25]), total=len(all_docs_for_bm25), desc="Lemmatizing Corpus"):
        lemmatized_tokens = [token.lemma_.lower() for token in spacy_doc if not token.is_punct and not token.is_space]
        original_metadata = all_docs_for_bm25[len(lemmatized_docs)].metadata
        lemmatized_docs.append(Document(page_content=" ".join(lemmatized_tokens), metadata=original_metadata))

    logger.info("Creating BM25 index...")
    bm25_retriever = BM25Retriever.from_documents(documents=lemmatized_docs)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(bm25_retriever, f)
    logger.info(f"New BM25 Retriever built and saved to {save_path}")
    