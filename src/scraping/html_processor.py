import logging
import re
from bs4 import BeautifulSoup, Tag
from html2text import HTML2Text
from pathlib import Path

from .agent import TableLinearizer

# --- Logger Setup ---

table_logger = logging.getLogger('TableProcessor')
table_logger.setLevel(logging.WARNING)
# if setup_table_logger isn't called.
if not table_logger.handlers:
    table_logger.addHandler(logging.NullHandler())

def setup_table_logger(log_dir_path: Path):
    """Configures the table logger to write warnings and errors to a file."""
    log_dir_path.mkdir(parents=True, exist_ok=True) # Ensure log directory exists
    table_log_file = log_dir_path / "table_warnings.log"

    # Remove existing handlers (like NullHandler) before adding the file handler
    for handler in table_logger.handlers[:]:
        table_logger.removeHandler(handler)
        handler.close() # Close handler before removing

    file_handler = logging.FileHandler(table_log_file, encoding='utf-8')
    file_handler.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - Doc: %(doc_url)s - TableIdx: %(table_idx)s - %(message)s')
    file_handler.setFormatter(formatter)
    table_logger.addHandler(file_handler)
    
    # Use the logger instance itself to log the setup confirmation
    table_logger.info(f"TableProcessor logger configured. Warnings/errors will be logged to: {table_log_file}")

# --- Helper Functions ---

def _collapse_newlines(text: str, max_consecutive: int = 2, keep: int = 2) -> str:
    """
    Collapse sequences of more than `max_consecutive` consecutive newlines
    into exactly `keep` newlines. Internal function used for post-html2text cleanup.
    
    Args:
        text (str): Input text.
        max_consecutive (int): Number of consecutive newlines above which collapsing occurs.
        keep (int): Number of newlines to replace the collapsed sequence with.

    Returns:
        str: The text with collapsed newlines.
    """
    # Pattern: match `max_consecutive + 1` or more consecutive newlines
    pattern = rf'\n{{{max_consecutive + 1},}}'
    return re.sub(pattern, '\n' * keep, text.strip())

# --- Main Processing Function ---

def html_to_clean_markdown(html_content: str, table_linearizer: TableLinearizer, base_url: str = "") -> str:
    """
    Converts raw HTML content to cleaned markdown, handling tables by
    converting them to text/markdown and then linearizing with an LLM agent.
    
    Args:
        html_content: The raw HTML string.
        base_url: The base URL for resolving relative links within the HTML.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    main_content = soup.select_one("#sliph-document-content") or soup

    # --- Remove "Top of Page" links ---
    for top_link in main_content.find_all("span", class_="top-link"):
        # Check if it contains the specific link text for safety
        if top_link.find("a", href="#document-top", string="Top of Page"):
            top_link.decompose() # Remove the entire span tag

    # --- Table Processing ---
    h_converter = HTML2Text(baseurl=base_url)
    h_converter.ignore_links = False # Ensure links are processed
    h_converter.body_width = 0       # Prevent line wrapping

    for table_index, table in enumerate(main_content.find_all("table")):
        log_extra = {'doc_url': base_url, 'table_idx': table_index + 1}
        try:
            # 1. Convert the raw table HTML to Markdown/Text
            table_markdown = h_converter.handle(str(table)).strip()            

            # Skip if conversion results in empty string
            if not table_markdown:
                table.decompose()
                continue

            # 2. Linearize the Markdown/Text representation using the agent
            linearized_text = table_linearizer(table_markdown_text=table_markdown)

            if not linearized_text:
                # Use the dedicated table logger
                table_logger.warning(
                    f"Linearizer returned empty result. Input snippet: {table_markdown[:200]}...",
                    extra=log_extra
                )
                table.decompose()
                continue
            
            # 3. Create replacement list
            replacement_ul = soup.new_tag("ul")
            for sentence in linearized_text.split('\n'):
                if sentence.strip():
                    li_tag = soup.new_tag("li")
                    li_tag.string = sentence.strip().lstrip('-* ')
                    replacement_ul.append(li_tag)
            
            # Replace the original <table> tag
            table.replace_with(replacement_ul)
            # print(f"  - Replaced a table with a linearized list.")

        except Exception as e:
            # Use the dedicated table logger for errors too
            table_logger.error(
                f"Processing failed: {e}. Table snippet: {str(table)[:200]}...",
                extra=log_extra
            )
            table.decompose()

    # --- Main HTML to Markdown Conversion ---    
    markdown = h_converter.handle(str(main_content))

    # --- Final Cleanup ---
    final_markdown = _collapse_newlines(markdown)
    
    return final_markdown