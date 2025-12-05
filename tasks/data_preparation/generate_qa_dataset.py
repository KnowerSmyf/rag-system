import dspy
import random
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from typing import List, Dict, Any, Tuple

# Import project modules
from src.data_processing import qa_generation
from src.data_processing.utils import load_generation_data
from src.utils import get_active_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _parse_arguments(cfg) -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate synthetic QA dataset.")
    parser.add_argument("--num_single_hop", type=int, default=cfg.NUM_SINGLE_HOP,
                        help=f"Number of single-hop examples (default: {cfg.NUM_SINGLE_HOP})")
    parser.add_argument("--groups_file", type=Path, default=cfg.MANUAL_GROUPS_FILE,
                        help=f"Path to curated document groups JSON (default: {cfg.MANUAL_GROUPS_FILE})")
    parser.add_argument("--test_size", type=int, default=cfg.TEST_SET_SIZE,
                        help=f"Number of test examples (default: {cfg.TEST_SET_SIZE})")
    parser.add_argument("--num_distractors", type=int, default=1,
                        help="Number of distractor documents to add (default: 1)")
    return parser.parse_args()

def _setup_environment(cfg):
    """Loads environment variables and configures DSPy LMs."""
    logger.info("Setting up QA Generation Environment...")
    load_dotenv()
    try:
        # NOTE Anthropic models are used for QA generation to reduce model-as-evaluator biases 
        # during evaluation of the RAG system (which uses Gemini).
        teacher_lm = dspy.LM(cfg.TEACHER_LM_MODEL, **cfg.TEACHER_LM_KWARGS)
        classifier_lm = dspy.LM(cfg.CLASSIFIER_LM_MODEL, **cfg.CLASSIFIER_LM_KWARGS)
        dspy.configure(lm=classifier_lm, teacher_lm=teacher_lm)
        logger.info("Models configured.")
        return True, dspy.settings.get('lm'), dspy.settings.get('teacher_lm')
    except Exception as e:
        logger.error(f"Failed to configure Anthropic models: {e}", exc_info=True)
        return False, None, None


def _generate_examples(
    generator: qa_generation.SyntheticDataGenerator, 
    all_documents: List[Dict], 
    doc_map: Dict[str, Dict], 
    document_groups: List[List[str]], 
    num_single_hop: int
) -> List[Dict[str, Any]]:
    """Runs the single-hop and multi-hop generation loops."""
    generated_examples = []
    
    # --- SINGLE-HOP ---
    logger.info(f"Generating {num_single_hop} single-hop examples...")
    generated_count = 0
    attempts = 0
    max_attempts = num_single_hop * 2 # Safety break for consistent failures
    
    with tqdm(total=num_single_hop, desc="Generating Single-Hop") as pbar:
        while generated_count < num_single_hop and attempts < max_attempts:
            doc = random.choice(all_documents)
            attempts += 1
            try:
                # generate_qa_pair expects a list
                example = generator.generate_qa_pair(documents=[doc]) 
                generated_examples.append(example)
                generated_count += 1
                pbar.update(1)
            except Exception as e:
                logger.warning(f"Skipping single-hop example due to error: {e}")

    if generated_count < num_single_hop:
        logger.warning(f"Only generated {generated_count}/{num_single_hop} single-hop examples after {max_attempts} attempts.")
    
    # --- MULTI-HOP ---
    logger.info(f"Generating multi-hop examples for {len(document_groups)} curated groups...")
    multi_hop_generated_count = 0
    for group_titles in tqdm(document_groups, desc="Generating Multi-Hop"):
        group_docs = []
        valid_group = True
        for title in group_titles:
            doc = doc_map.get(title.lower())
            if doc:
                group_docs.append(doc)
            else:
                logger.warning(f"Title '{title}' from group not found in corpus map. Skipping group: {group_titles}")
                valid_group = False
                break
        
        if valid_group and len(group_docs) > 1:
            try:
                example = generator.generate_qa_pair(documents=group_docs)
                generated_examples.append(example)
                multi_hop_generated_count += 1
            except Exception as e:
                logger.warning(f"Skipping multi-hop example for group {group_titles} due to error: {e}")
    
    logger.info(f"Generated {multi_hop_generated_count} multi-hop examples.")
    
    return generated_examples

def main(config_module=None) -> None:
    """
    Generates a synthetic QA dataset for RAG evaluation.
    - Generates single-hop examples based on random documents.
    - Generates multi-hop examples based on manually curated document groups.
    - Adds distractor documents.
    - Splits into train/test sets and saves as JSONL.
    """
    cfg = get_active_config(config_module)
    args = _parse_arguments(cfg)

    success, _, teacher_lm = _setup_environment(cfg)
    if not success: return

    # Use the outsourced data loading function
    all_documents, doc_map, document_groups = load_generation_data(
        cfg.CORPUS_FILE, args.groups_file
    )
    
    generator = qa_generation.SyntheticDataGenerator(teacher_lm=teacher_lm)

    # Call the internal helper for generation loops
    generated_examples = _generate_examples(
        generator, all_documents, doc_map, document_groups, args.num_single_hop
    )

    # Call the outsourced processing and saving function
    qa_generation.process_and_save_dataset(
        generated_examples, 
        all_documents, 
        cfg.SYNTHETIC_DATASET_DIR, 
        args.test_size,
        args.num_distractors # Pass num_distractors from args
    )

    logger.info("Dataset generation complete!")

if __name__ == "__main__":
    main()