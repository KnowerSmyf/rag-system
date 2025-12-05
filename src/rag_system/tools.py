from .signatures import GenerateQueriesSignature, GenerateHypotheticalAnswer
import dspy
import spacy
from dspy import Tool
from typing import List, Dict

from src.data_processing.utils import load_corpus

class HybridSearch(dspy.Module):
    """A module that performs both multi-query expansion and HyDE."""
    def __init__(self):
        super().__init__()
        # Use a faster LM for these simple generation tasks
        self.query_expander = dspy.Predict(GenerateQueriesSignature) # , lm=gemini_flash)
        self.hyde_generator = dspy.Predict(GenerateHypotheticalAnswer) # , lm=gemini_flash)

    def forward(self, query):
        expanded = self.query_expander(query=query).expanded_queries
        hypothetical = self.hyde_generator(question=query).hypothetical_answer
        # The final search list includes the original query, expansions, and the HyDE answer
        all_search_texts = [query] + expanded + [hypothetical]
        return dspy.Prediction(search_queries=all_search_texts)

class PPLTools:
    """A class to encapsulate all data stores and tool functions for the PPL RAG system."""
    def __init__(self, vectorstore, metastore, bm25_retriever, query_lm, corpus_file_path):
        self.vectorstore = vectorstore
        self.metastore = metastore
        self.bm25_retriever = bm25_retriever
        self.nlp = spacy.load("en_core_web_sm")
        self.hybrid_search = HybridSearch()
        self.hybrid_search.set_lm(query_lm)
        print("✅ PPLTools initialized with all necessary data stores.")

        # --- Load corpus into memory for fast title lookup ---
        self._corpus_map = self._load_corpus_map(corpus_file_path)

    def _load_corpus_map(self, corpus_path) -> Dict[str, str]:
        """Loads the JSONL corpus and creates a title -> text dictionary."""
        corpus_data = load_corpus(corpus_path) # Use the helper from utils
        corpus_map = {}
        for doc in corpus_data:
            title = doc.get("title")
            text = doc.get("text")
            if title and text:
                # Store text by lowercased title for case-insensitive matching
                corpus_map[title.lower()] = text 
        return corpus_map

    def semantic_content_search(self, query: str) -> str:
        # Use the hybrid search module to get expanded queries
        queries = self.hybrid_search(query=query).search_queries
        
        all_docs = []
        doc_ids = set()
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

        for q in queries:
            docs = retriever.invoke(q)
            for doc in docs:
                doc_id = doc.metadata.get('title', '') + doc.page_content[:100]
                if doc_id not in doc_ids:
                    all_docs.append(doc)
                    doc_ids.add(doc_id)
        
        return "\n\n".join([f"Source: {doc.metadata.get('title', 'Unknown')}\nContent: {doc.page_content}" for doc in all_docs]) if all_docs else f"No content found for '{query}'."

    def keyword_search(self, query: str) -> str:
        spacy_doc = self.nlp(query)
        lemmatized_tokens = [token.lemma_.lower() for token in spacy_doc if not token.is_punct and not token.is_space]
        lemmatized_query = " ".join(lemmatized_tokens)
        result_docs = self.bm25_retriever.invoke(lemmatized_query, k=5)
        return "\n\n".join([f"Source: {doc.metadata.get('title', 'Unknown')}\nContent: {doc.page_content}" for doc in result_docs]) if result_docs else f"No documents found containing the keywords: '{query}'."

    def get_full_document_text(self, title: str) -> str:
        """
        Retrieves the COMPLETE text of a single document given its exact title,
        using an in-memory map of the corpus. Case-insensitive.
        """
        title = title.strip().strip('\'"').lower()
        # Retrieve text from the in-memory map
        document_text = self._corpus_map.get(title)
        if document_text:
            return document_text
        else:
            return f"Error: No document found with the title '{title}'."
        
    def semantic_metadata_search(self, query: str) -> List[str]:
        results = self.metastore.similarity_search(query, k=5)
        unique_titles = list(dict.fromkeys([doc.metadata['title'] for doc in results if 'title' in doc.metadata]))
        return unique_titles if unique_titles else [f"No relevant documents found for '{query}' based on their titles and descriptions."]

def get_final_agentic_tools(tool_functions: PPLTools) -> List[Tool]:
    """Initializes and returns the final list of dspy.Tool objects."""
    tools = [
        Tool(name="semantic_metadata_search", func=tool_functions.semantic_metadata_search, desc="Performs a fast semantic search on only the titles and descriptions of documents. Use this as a FIRST STEP to quickly identify a list of potentially relevant policies based on a general topic or concept. Returns a list of document titles and descriptions."),
        Tool(name="semantic_content_search", func=tool_functions.semantic_content_search, desc="Performs a broad semantic search on the FULL TEXT content of all documents. Use this to find specific facts, procedures, or nuanced details deep inside documents, especially for vague questions where the exact keywords are unknown. Returns relevant text chunks and their source titles."),
        Tool(name="keyword_search", func=tool_functions.keyword_search, desc="Searches the full text of all documents for exact keyword matches. Best for finding specific, non-negotiable terms like official form names, acronyms, or technical terms that semantic search might misinterpret. Use multiple comma-separated keywords to broaden the search and find documents that may not use the exact original term (e.g., 'AI, artificial, intelligence, generative')."),
        Tool(name="get_full_document_text", func=tool_functions.get_full_document_text, desc="Retrieves the COMPLETE, unabridged text of a single document given its exact title. Use this only AFTER you have identified a highly relevant document title from one of your search tools to gain a comprehensive understanding."),
    ]
    return tools
