import logging
import json
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import dspy

# Configure logger
logger = logging.getLogger(__name__)

def cuda_device() -> str:
    """Checks for CUDA availability and returns 'cuda' or 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_corpus(corpus_path: Path) -> List[Dict]:
    """Loads all documents from the JSONL corpus."""
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found at: {corpus_path}")
    
    documents = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            documents.append(json.loads(line))
    return documents

def load_dspy_dataset(file_path: Path, fields_to_load: Optional[List[str]] = None) -> List[dspy.Example]:
    """
    Loads a .jsonl file into dspy.Example objects.

    Args:
        file_path: Path to the .jsonl file.
        fields_to_load: If provided, only these keys will be loaded into the Example.
                        Otherwise, all keys are loaded.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
        
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            
            if fields_to_load:
                # Load only specified fields
                example_data = {k: data.get(k) for k in fields_to_load if k in data}
            else:
                # Load all fields
                example_data = data

            # Ensure 'question' is always present for with_inputs
            if 'question' not in example_data:
                logger.warning(f"Skipping line, 'question' field missing: {line.strip()}")
                continue

            # Use dictionary unpacking to load all fields
            example = dspy.Example(**example_data).with_inputs("question")
            
            # Standardize the 'gold_answer' field if 'response' was used
            if not fields_to_load and hasattr(example, 'response') and not hasattr(example, 'gold_answer'):
                example.gold_answer = example.response

            examples.append(example)

    print(f"✅ Loaded {len(examples)} examples from {file_path}")
    return examples

def load_document_groups(file_path: Path) -> List[List[str]]:
    """
    Loads manually curated groups of document titles from a JSON file.

    Expects a JSON file containing a list of lists of strings (document titles).

    Args:
        file_path: The path to the JSON file containing the groups.

    Returns:
        A list of lists, where each inner list contains the titles for a group.

    Raises:
        FileNotFoundError: If the specified file_path does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        TypeError: If the loaded JSON is not a list of lists of strings.
    """
    logger.info(f"Loading manually curated document groups from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            groups = json.load(f)

        # --- Input Validation ---
        if not isinstance(groups, list):
            raise TypeError(f"Expected a list of groups, but got {type(groups)}.")
        
        for i, group in enumerate(groups):
            if not isinstance(group, list):
                raise TypeError(f"Group at index {i} is not a list, but {type(group)}.")
            if not all(isinstance(title, str) for title in group):
                raise TypeError(f"Group at index {i} contains non-string elements.")

        logger.info(f"Successfully loaded {len(groups)} document groups.")
        return groups

    except FileNotFoundError:
        logger.error(f"Document groups file not found at '{file_path}'.")
        raise # Re-raise the specific error
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from '{file_path}': {e}")
        raise # Re-raise the specific error
    except TypeError as e:
        logger.error(f"Invalid format in '{file_path}': {e}")
        raise # Re-raise the specific error

def load_generation_data(corpus_path: Path, groups_path: Path) -> Tuple[List[Dict], Dict[str, Dict], List[List[str]]]:
    """
    Loads all necessary data for QA generation: the full corpus, a title-to-doc map,
    and curated multi-hop document groups.

    Args:
        corpus_path: Path to the scraped_corpus.jsonl file.
        groups_path: Path to the manual_document_groups.json file.

    Returns:
        A tuple containing:
            - all_documents (List[Dict]): The full corpus.
            - doc_map (Dict[str, Dict]): Lowercase title to document mapping.
            - document_groups (List[List[str]]): List of title groups.
            
    Raises:
        FileNotFoundError: If the corpus_path does not exist.
    """
    logger.info(f"Loading corpus from {corpus_path}...")
    all_documents = load_corpus(corpus_path) # Assumes load_corpus raises FileNotFoundError
    doc_map = {doc['title'].lower(): doc for doc in all_documents if 'title' in doc}
    logger.info(f"Created lookup map for {len(doc_map)} documents.")
    
    document_groups = []
    try:
        document_groups = load_document_groups(groups_path)
    except Exception as e:
        # Log warning but don't stop execution, allow single-hop generation.
        logger.warning(f"Could not load or validate document groups from '{groups_path}': {e}. Multi-hop generation will be skipped.")
        
    return all_documents, doc_map, document_groups
