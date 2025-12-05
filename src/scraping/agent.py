"""
Defines a DSPy module for converting structured table data (in JSON form)
into linearized, self-contained factual sentences. This is useful for
scraping pipelines that need to transform tabular content into natural
language facts for downstream processing.

Classes:
    LinearizeTableSignature: DSPy signature defining inputs and outputs.
    TableLinearizer: Module that performs the table-to-text transformation.

Module-level objects:
    table_linearizer: Singleton instance of TableLinearizer for reuse.
"""
import dspy
from dotenv import load_dotenv

# Ensure environment variables (e.g., GOOGLE_API_KEY) are loaded before model initialization
load_dotenv()

class LinearizeTableSignature(dspy.Signature):
    """
    Converts a text/markdown representation of a table into a list of 
    self-contained, factual sentences. Each sentence must incorporate 
    relevant context (like titles or headers) to make sense in isolation. 
    Preserve markdown links exactly.
    """
    
    __doc__ = """
    You are a meticulous data analyst preparing text for a research database. Your primary goal is to convert the provided text/markdown representation of a table into a list of sentences where each sentence is a **complete, self-contained fact**.

    **Instructions:**
    1.  Analyze the input text to understand the table's structure, including any titles (often found below the table) or multi-row headers.
    2.  For each logical data row, create one clear sentence.
    3.  **Context Integration:** In each sentence, incorporate necessary context (e.g., from column headers and the overall table title) so the sentence makes sense independently. If combining multi-row headers, ensure the resulting context is clear (e.g., "Category 1 Altitude"). If no title is obvious, use context from the primary header row.
    4.  **CRITICAL: Link Preservation:** Preserve any markdown-style links (e.g., `[link text](URL)`) exactly as they appear in the original text. Do not remove or alter them.
    5.  **Handle Non-Tabular Text:** If the input looks like a table but is actually just formatted paragraphs or lists (like a legend), output the text paragraphs directly, adding the title if one exists.
    6.  Do not add any summary or introductory text. Output only the factual sentences, one per line.

    ---
    **Example 1: Simple Table with Title Below**

    **Input (Table as Markdown/Text):**
    Maximum dive depth m | Maximum daily dive time (minutes)
    ---|---
    One dive only | Multiple dives
    6 | 480 | 360
    9 | 240 | 190
    >9 | 150 | 120
    *Table 1 – Maximum time limits for divers undertaken where recompression chamber support is available within 2 hours*

    **Output (Factual Sentences):**
    According to Table 1 (Maximum time limits for divers where recompression support is available within 2 hours), for a maximum dive depth of 6m, the maximum daily dive time is 480 minutes for one dive only and 360 minutes for multiple dives.
    According to Table 1 (Maximum time limits for divers where recompression support is available within 2 hours), for a maximum dive depth of 9m, the maximum daily dive time is 240 minutes for one dive only and 190 minutes for multiple dives.
    According to Table 1 (Maximum time limits for divers where recompression support is available within 2 hours), for a maximum dive depth >9m, the maximum daily dive time is 150 minutes for one dive only and 120 minutes for multiple dives.

    ---
    **Example 2: Multi-Row Header Table**

    **Input (Table as Markdown/Text):**
    Altitude (m) | Minimum delay before travel to altitude (h)
    ---|---
    Category of dive (see below legend)
    1 | 2 | 3
    0-150 | Nil | Nil | 2
    150-600 | Nil | 2 | 12
    600-2400 | 12 | 24 | 48
    *Table 75 – Minimum delay before exposure to altitude*

    **Output (Factual Sentences):**
    According to Table 75 (Minimum delay before exposure to altitude), for an altitude of 0-150m, the minimum delay is Nil hours for Category 1 dives, Nil hours for Category 2 dives, and 2 hours for Category 3 dives.
    According to Table 75 (Minimum delay before exposure to altitude), for an altitude of 150-600m, the minimum delay is Nil hours for Category 1 dives, 2 hours for Category 2 dives, and 12 hours for Category 3 dives.
    According to Table 75 (Minimum delay before exposure to altitude), for an altitude of 600-2400m, the minimum delay is 12 hours for Category 1 dives, 24 hours for Category 2 dives, and 48 hours for Category 3 dives.

    ---
    **Example 3: Definition Table with Links**

    **Input (Table as Markdown/Text):**
    Term | Definition
    ---|---
    BCD | Buoyancy Control Device; typically a vest worn by the diver...
    CCR | An [underwater breathing apparatus](URL1) that absorbs the [carbon dioxide](URL2)...

    **Output (Factual Sentences):**
    The term BCD is defined as: Buoyancy Control Device; typically a vest worn by the diver...
    The term CCR is defined as: An [underwater breathing apparatus](URL1) that absorbs the [carbon dioxide](URL2)...

    ---
    **Example 4: Non-Tabular Text (Legend)**

    **Input (Table as Markdown/Text):**
    Category 1: A single dive to <50% of the DCIEM no-decompression limit...
    Category 2: Dives exceeding category 1 but not included in Category 3...
    Category 3: Repetitive deep diving over multiple days...

    **Output (Factual Sentences):**
    Category 1: A single dive to <50% of the DCIEM no-decompression limit...
    Category 2: Dives exceeding category 1 but not included in Category 3...
    Category 3: Repetitive deep diving over multiple days...
    """
    table_markdown_text = dspy.InputField(desc="The text or markdown representation of the table.")
    linearized_sentences = dspy.OutputField(desc="A string where each line is a complete, self-contained factual sentence, preserving markdown links.")


class TableLinearizer(dspy.Module):
    """A DSPy module for converting HTML tables into a linear list of sentences."""
    def __init__(self, model_kwargs: dict = {}):
        super().__init__()
        # self.model = dspy.LM(**config.SCRAPING_HELPER_MODEL_KWARGS)
        self.model = dspy.LM(**model_kwargs)
        self.predictor = dspy.Predict(signature=LinearizeTableSignature)

    def forward(self, table_markdown_text: str):
        """
        Takes a text/markdown string of a table and returns linearized sentences.
        """
        with dspy.context(lm=self.model):
            prediction = self.predictor(table_markdown_text=table_markdown_text)
            return prediction.linearized_sentences
