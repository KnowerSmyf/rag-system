import os
import dspy
from typing import Optional, List
from pathlib import Path
from langchain.schema import Document
from sentence_transformers.cross_encoder import CrossEncoder

# Import from local modules
from .signatures import RetrieverSignature, SynthesiserSignature
from .tools import PPLTools # Assuming PPLTools might be needed here


class AgenticRAG(dspy.Module):
    def __init__(self, tools, retriever_path: Optional[Path] = None, synthesizer_path: Optional[Path] = None):
        super().__init__()
        self.get_full_text_func = [t.func for t in tools if t.name == "get_full_document_text"][0]
        
        # Initialize the modules first
        self.retriever = dspy.ReAct(RetrieverSignature, tools=tools)
        self.synthesiser = dspy.ChainOfThought(SynthesiserSignature)

        # Helper for checking paths
        def verify_path(path: Path, label: str):
            resolved = path.resolve()
            if not resolved.exists():
                print(
                    f"[AgenticRAG] ⚠️ The supplied {label} path does not exist.\n"
                    f"  Supplied: {path}\n"
                    f"  Resolved (from {Path(os.getcwd())}): {resolved}\n"
                    f"  Proceeding with an uncompiled {label} module."
                )
                return False
            else:
                print(f"[AgenticRAG] ✅ Loading compiled {label} state from: {resolved}")
                return True

        # Conditionally load retriever and synthesiser states
        if retriever_path and verify_path(retriever_path, "retriever"):
            self.retriever.load(str(retriever_path.resolve()))

        if synthesizer_path and verify_path(synthesizer_path, "synthesiser"):
            self.synthesiser.load(str(synthesizer_path.resolve()))


    def forward(self, question: str):
        retrieved_info = self.retriever(question=question)
        titles, notes = retrieved_info.titles, retrieved_info.notes

        full_text_contents = []
        for title in titles:
            src = f"Source Document: {title}\n\n"
            body = self.get_full_text_func(title)
            full_text_contents.append(src + body)

        # Stitch together the final context for the synthesiser.
        context = "--- Source Documents ---\n"
        context += "\n\n---\n\n".join(full_text_contents)
        context += "\n\n--- Researcher's Notes ---\n"
        context += notes

        # Call the synthesiser to generate the final answer.
        prediction = self.synthesiser(question=question, context=context)
        return dspy.Prediction(answer=prediction.answer, titles=titles)


class CrossEncoderReRanker:
    """A non-LLM re-ranker using a Cross-Encoder model."""
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model = CrossEncoder(model_name)
        print(f"✅ Cross-Encoder model '{model_name}' loaded successfully.")

    def __call__(self, query: str, chunks: List[Document]) -> List[Document]:
        if not chunks:
            return []
        model_input_pairs = [[query, chunk.page_content] for chunk in chunks]
        scores = self.model.predict(model_input_pairs)
        chunk_with_scores = list(zip(chunks, scores))
        sorted_chunks = sorted(chunk_with_scores, key=lambda x: x[1], reverse=True)
        return [chunk for chunk, score in sorted_chunks]

class VanillaRAG(dspy.Module):
    """A 'steel-manned' traditional RAG pipeline with a re-ranker."""
    def __init__(self, vectorstore, reranker, top_k_rerank=5):
        super().__init__()
        self.retriever = dspy.Retrieve(k=20) # Retrieve a large number of candidates
        self.reranker = reranker
        self.synthesizer = dspy.ChainOfThought(SynthesiserSignature)
        self.vectorstore = vectorstore
        self.top_k_rerank = top_k_rerank

    def forward(self, question):
        retrieved_chunks = self.vectorstore.similarity_search(query=question, k=20)
        reranked_chunks = self.reranker(query=question, chunks=retrieved_chunks)
        top_chunks = reranked_chunks[:self.top_k_rerank]
        top_chunks_content = [
            f"Title: {chunk.metadata.get('title', 'N/A')}\nContent: {chunk.page_content}"
            for chunk in top_chunks
        ]
        context = "\n\n---\n\n".join(top_chunks_content)
        prediction = self.synthesizer(question=question, context=context)
        retrieved_titles = [chunk.metadata.get('title', 'N/A') for chunk in top_chunks]
        return dspy.Prediction(
            answer=prediction.answer, 
            titles=list(dict.fromkeys(retrieved_titles))
        )
