import json
import time
import argparse
import logging
from tqdm import tqdm

# Import project modules
from src.scraping import metadata_fetcher, html_processor, utils, agent
from src.utils import get_active_config

# --- Configure Logging ---
# Set up basic logging to capture INFO level messages and above
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(limit: int, config_module=None):
    """
    Executes the full scraping and processing pipeline.
    1. Fetches metadata for all documents.
    2. Scrapes the HTML for each document.
    3. Converts HTML to clean markdown in memory, linearizing tables.
    4. Saves the final, structured data to a single JSONL file.
    """
    cfg = get_active_config(config_module)

    # --- Define Error Log Path ---
    error_log_path = cfg.LOG_DIR / "scraping_errors.log"

    # --- Safety Check ---
    if cfg.CORPUS_FILE.exists() and not limit:
        overwrite = input(f"'{cfg.CORPUS_FILE}' already exists. A full re-scrape will overwrite it. Continue? (y/N): ").lower()
        if overwrite != 'y':
            logger.info("Exiting without overwriting.")
            return

    logger.info("--- Starting Full PPL Scraping Pipeline ---")

    # Set up the dedicated logger for table processing warnings/errors
    try:
        html_processor.setup_table_logger(cfg.LOG_DIR)
    except Exception as e:
        logger.error(f"Failed to set up table logger: {e}", exc_info=True)
        # Continue execution even if logger setup fails, but log the error

    session = utils.setup_session()

    # --- Stage 1: Get Metadata ---
    try:
        all_metadata = metadata_fetcher.get_all_metadata(session, cfg.PPL_BROWSE_URL)
        if not all_metadata:
            logger.error("No metadata fetched. Exiting.")
            return
    except Exception as e:
        logger.error(f"Failed during metadata fetching: {e}", exc_info=True)
        return

    if limit:
        all_metadata = all_metadata[:limit]
        logger.warning(f"Limiting scrape to the first {limit} documents.")

    # --- Instantiate the Table Linearizer ---
    try:
        table_linearizer = agent.TableLinearizer(
            model_kwargs=cfg.SCRAPING_HELPER_MODEL_KWARGS
        )
        logger.info("TableLinearizer module initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize TableLinearizer: {e}", exc_info=True)
        return

    # --- Stage 2: Scrape, Process, and Save Full Pages ---
    processed_count = 0
    # Ensure error log directory exists
    cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
    # Clear previous error log if forcing full rescrape
    if cfg.CORPUS_FILE.exists() and not limit:
        if error_log_path.exists():
            logger.info(f"Clearing previous error log: {error_log_path}")
            error_log_path.unlink()

    with open(cfg.CORPUS_FILE, 'w', encoding='utf-8') as f:
        for index, meta in enumerate(tqdm(all_metadata, desc="Scraping and Processing Documents")):
            current_doc_identifier = f"Document {index+1}/{len(all_metadata)}: '{meta.get('title', meta['url'])}'"
            logger.info(f"Processing {current_doc_identifier}...")

            try:
                # Scrape the full HTML page
                resp = session.get(meta['url'])
                resp.raise_for_status()

                # Process the HTML to clean markdown, passing the linearizer instance
                markdown_text = html_processor.html_to_clean_markdown(
                    resp.text,
                    table_linearizer=table_linearizer, # Pass the instance
                    base_url=meta['url']
                )

                # Combine metadata and text
                record = {
                    "url": meta['url'],
                    "title": meta['title'],
                    "description": meta.get('summary', ''),
                    "text": markdown_text
                }

                # Write the record
                f.write(json.dumps(record) + '\n')
                processed_count += 1

                # Apply polite throttling
                time.sleep(cfg.SCRAPING_THROTTLE_SECONDS)

            except Exception as e:
                logger.error(f"Failed to process {current_doc_identifier}: {e}", exc_info=True)
                # Log the failed URL to the error file
                with open(error_log_path, "a", encoding="utf-8") as err_f:
                    err_f.write(f"{meta['url']}\t{e}\n")

    logger.info(f"\nPipeline complete! Successfully processed {processed_count}/{len(all_metadata)} documents.")
    # Report if there were errors
    if error_log_path.exists():
        with open(error_log_path, 'r') as err_f:
             error_count = sum(1 for _ in err_f)
        if error_count > 0:
            logger.warning(f"{error_count} documents failed to process. Check '{error_log_path.resolve()}' for details.")
        else:
            # Optionally remove empty log file
            error_log_path.unlink()


if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run the full PPL scraping and processing pipeline.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of documents to scrape for testing (e.g., --limit 5).")
    args = parser.parse_args()
    limit = args.limit # Get the limit value

    # Call main directly. It handles args and config loading internally.
    main(limit=args.limit)