import os
import shutil
import argparse # Add argparse for flags
from pathlib import Path
import logging
import json

import test_config as cfg

# --- Import task functions ---
from tasks.data_preparation import (
    build_agentic_db, build_metadata_db, build_text_split_db, generate_qa_dataset
)
from tasks.training import compile_program
from tasks.evaluation import run_evaluation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Define functions for each stage ---

def clean_outputs():
    """Cleans test output directories."""
    output_dirs = [cfg.STORES_DIR, cfg.SYNTHETIC_DATASET_DIR, cfg.RESULTS_DIR, cfg.LOG_DIR]
    logger.info("--- Cleaning previous test outputs ---")
    for dir_path in output_dirs:
        if dir_path.exists():
            logger.info(f"Removing directory: {dir_path}")
            shutil.rmtree(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

def run_data_prep():
    """Runs all data preparation tasks."""
    logger.info("\n--- Running Data Preparation ---")
    build_metadata_db.main(config_module=cfg)
    assert cfg.METADATA_DB_PATH.exists(), "Metadata DB dir missing."
    build_text_split_db.main(config_module=cfg, force_rebuild=True)
    assert cfg.TEXTSPLIT_DB_PATH.exists(), "TextSplit DB dir missing."
    build_agentic_db.main(config_module=cfg)
    assert cfg.AGENTIC_DB_PATH.exists(), "Agentic DB dir missing."
    assert cfg.BM25_PATH.exists(), "BM25 Cache file missing."
    assert cfg.CHUNK_CACHE_FILE.exists(), "Chunk Cache file missing."
    logger.info("✅ Data preparation tasks completed successfully.")

def run_synth_data():
    """Runs synthetic data generation."""
    logger.info("\n--- Running Synthetic Data Generation ---")
    # Assuming generate_qa_dataset takes relevant args or uses test_config defaults
    generate_qa_dataset.main(config_module=cfg)
    assert (cfg.SYNTHETIC_DATASET_DIR / "train.jsonl").exists(), "Train set missing."
    assert (cfg.SYNTHETIC_DATASET_DIR / "test.jsonl").exists(), "Test set missing."
    logger.info("✅ Synthetic data generation completed successfully.")

def run_training():
    """Runs agent compilation."""
    logger.info("\n--- Running Agent Compilation ---")
    compile_program.main(config_module=cfg)
    assert cfg.COMPILED_AGENT_PATH.exists(), "Compiled agent file missing."
    logger.info("✅ Agent compilation completed successfully.")

def run_eval():
    """Runs evaluation."""
    logger.info("\n--- Running Evaluation ---")
    run_evaluation.main(config_module=cfg)
    assert (cfg.RESULTS_DIR / "final_evaluation_detailed.csv").exists(), "Detailed results missing."
    assert (cfg.RESULTS_DIR / "final_evaluation_summary.csv").exists(), "Summary results missing."
    logger.info("✅ Evaluation run completed successfully.")

def test_agentic_cache_partial():
    """Tests agentic chunking with a partially deleted cache."""
    logger.info("\n--- Testing Agentic Caching (Partial) ---")
    if not cfg.CHUNK_CACHE_FILE.exists():
        logger.error("Chunk cache file not found. Run full data prep first.")
        return

    # 1. Modify the cache (e.g., delete the last entry)
    try:
        with open(cfg.CHUNK_CACHE_FILE, 'r+', encoding='utf-8') as f:
            chunks = json.load(f)
            if not chunks:
                logger.warning("Cache is empty, cannot test partial caching.")
                return
            removed_item = chunks.pop() # Remove the last chunk entry
            f.seek(0)
            json.dump(chunks, f, indent=2)
            f.truncate()
        logger.info(f"Temporarily removed one entry (source: {removed_item.get('metadata',{}).get('title')}) from cache.")
    except Exception as e:
        logger.error(f"Failed to modify cache file: {e}")
        return

    # 2. Re-run agentic build (should only process the missing doc)
    logger.info("Re-running build_agentic_db with partial cache...")
    build_agentic_db.main(config_module=cfg) # This will check cache internally
    # Manually check logs for confirmation it only processed 1 document.
    logger.info("✅ Partial cache test finished. Check logs to verify.")


def run_tests(args):
    """Orchestrates the dry run based on command line args."""
    logger.info("--- Starting Dry Run ---")

    if args.clean_first:
        clean_outputs()

    # --- Run Stages Conditionally ---
    if not args.skip_prep:
        try:
            run_data_prep()
        except Exception as e:
            logger.error(f"❌ ERROR during data preparation stage: {e}", exc_info=True)
            return # Stop

    if args.test_cache_partial:
        try:
            test_agentic_cache_partial()
        except Exception as e:
            logger.error(f"❌ ERROR during partial cache test: {e}", exc_info=True)
            # Continue to next stages if desired, or return

    # Test full cache implicitly by running data prep again *without* cleaning
    if args.test_cache_full:
        logger.info("\n--- Testing Agentic Caching (Full) ---")
        logger.info("Re-running build_agentic_db with existing cache...")
        try:
            build_agentic_db.main(config_module=cfg) # Should print "All documents are already chunked"
            logger.info("✅ Full cache test finished. Check logs.")
        except Exception as e:
            logger.error(f"❌ ERROR during full cache test: {e}", exc_info=True)


    if not args.skip_synth:
        try:
            run_synth_data()
        except Exception as e:
            logger.error(f"❌ ERROR during synth data stage: {e}", exc_info=True)
            return

    if not args.skip_training:
        try:
            run_training()
        except Exception as e:
            logger.error(f"❌ ERROR during training stage: {e}", exc_info=True)
            return

    if not args.skip_eval:
        try:
            run_eval()
        except Exception as e:
            logger.error(f"❌ ERROR during evaluation stage: {e}", exc_info=True)
            return

    logger.info("\n--- Dry Run Script Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run dry run tests for the RAG pipeline.")
    parser.add_argument("--clean-first", action="store_true", help="Clean output directories before running.")
    parser.add_argument("--skip-prep", action="store_true", help="Skip the data preparation stage.")
    parser.add_argument("--skip-synth", action="store_true", help="Skip synthetic data generation.")
    parser.add_argument("--skip-training", action="store_true", help="Skip agent compilation.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip the evaluation stage.")
    parser.add_argument("--test-cache-full", action="store_true", help="Run agentic build twice to test full cache.")
    parser.add_argument("--test-cache-partial", action="store_true", help="Modify cache and re-run agentic build.")

    args = parser.parse_args()
    run_tests(args)



