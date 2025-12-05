import dspy

class GenerateQueriesSignature(dspy.Signature):
    """You are a search query expansion expert for the University of Queensland's Policy and Procedure Library.
    Your task is to rewrite a user's query into 3 diverse and effective queries to maximize search recall in a vector database.

    Instructions:
    1. Identify the intent and core concepts related to university policy, safety, or administration.
    2. Remove any irrelevent terms or details from the question, considering how it relates to policy, safety, or administration
    3. Generate a mix of query types: natural language questions, keyword searches, and semantic phrases.
    4. Add relevant synonyms and domain-specific policy terminology. 
    5. CRITICAL: If the original query contains a very specific term (e.g., 'dendrimers', 'isoflurane'), you MUST include that exact term in at least one of the generated queries.

    Provide only a numbered list of the 3 optimized search queries without any explanations, greetings, or additional commentary. For example:

    Query: "can I use AI for my math assignment?"
    Response:
    ["student use of generative artificial intelligence software tools for math assessment completion assistance", "AI assignment assistance tools automation generation completion ethics academic integrity plagiarism detection large language models machine learning natural language processing deep learning", What is the policy on using generative AI like large langage models (LLMs) for math assignments and coursework?"]
    """
    query = dspy.InputField()
    expanded_queries: list[str] = dspy.OutputField(desc="A Python list of 3 string queries.")

class GenerateHypotheticalAnswer(dspy.Signature):
    """Given a user's question, generate a detailed, hypothetical answer that contains the type of information you would expect to find in a relevant policy document."""
    question = dspy.InputField()
    hypothetical_answer: str = dspy.OutputField(desc="A detailed, paragraph-long hypothetical answer.")
    
class RetrieverSignature_v1(dspy.Signature):
    """Conduct research to find all policy document titles relevant to answering the user's question. Use your tools strategically. Start with precise keyword searches for specific terms, then broaden your search with semantic queries if needed. Make sure you are meticulous in your research. If one approach doesn't yield useful results, try something else. Your final output is a list of the most relevant document titles that will help answer the user's question."""
    question: str = dspy.InputField(desc="The user's original question.")
    titles: list[str] = dspy.OutputField(desc="A list of source document titles relevent to answering the question.")
    notes: str = dspy.OutputField(desc="Record any important findings or nuances about your observations that are not captured in the final list of documents.")

class RetrieverSignature(dspy.Signature):
    """Conduct research to find all policy document titles relevant to answering the user's question.

    **First, identify all the distinct concepts or topics in the user's query (e.g., different materials, separate procedures, etc.).**

    Your research strategy must **ensure you find relevant documents for *each* of these distinct concepts.** A single search for all terms at once may fail to find everything.

    Use your tools strategically. Start with precise keywords. **If a combined search doesn't return results for a specific topic, you MUST conduct a new, targeted search for that missing topic individually.**

    Be meticulous in your research. Before outputting your list, **verify that your collected titles actually cover all the key topics from the user's question.** Your final output is a list of the most relevant document titles that, together, will help answer the *entire* question."""
    question: str = dspy.InputField(desc="The user's original question.")
    titles: list[str] = dspy.OutputField(desc="A list of source document titles relevent to answering the question.")
    notes: str = dspy.OutputField(desc="Record any important findings or nuances about your observations that are not captured in the final list of documents.")

class SynthesiserSignature(dspy.Signature):
    """Use the contextual information to provide a comprehensive, clear, and authoritative answer to the user's question. Adhere strictly to the facts provided in the context."""
    question: str = dspy.InputField(desc="The user's original question.")
    context: str = dspy.InputField(desc="The concatenated full text of all relevant source documents, supplemented by any important notes from the research process")
    answer: str = dspy.OutputField(desc="A comprehensive and authoritative answer to the user's question, based strictly on the provided context.")
