"""
Reusable modules and signatures for generating a synthetic Question-Answering dataset.

This module contains the core DSPy components for creating realistic, multi-hop
question-answer pairs based on a corpus of policy documents. It includes logic
for persona generation, single and multi-hop QA synthesis, and post-processing
helpers to create a final, clean dataset.
"""
import logging
from tqdm import tqdm
import dspy
import json
import random
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__) # Add logger

# --- Personas ---
USER_PERSONAS = [
    "A first-year undergraduate student new to the university.",
    "A PhD candidate conducting complex laboratory research.",
    "A faculty member designing a new course curriculum.",
    "An administrative staff member managing university resources.",
    "A concerned parent asking about student safety and wellbeing.",
    "A visiting researcher from another institution.",
    "An IT staff member concerned about cyber security compliance.",
    "A researcher-entrepreneur interested in commercializing their university research through a start-up company."
]

# --- Signatures ---
class GenerateSingleHopQA(dspy.Signature):
    """
    Given the full text of a university policy document and a user persona, your task is to act as that user and generate one realistic, high-quality question that can be answered *only* using the provided text. Then, provide a comprehensive, gold-standard answer.

    ---
    **Crucial Guidelines for the Question:**
    - **Focus on a single, plausible user scenario.** The question should be grounded in a real-world problem a person might face.
    - **Be concise and realistic.** The question should sound like something a real person would type or ask.
    - **The goal is REALISM, not maximum logical complexity.**
    ---

    The answer must be detailed, authoritative, and strictly grounded in the provided document text.
    """
    persona = dspy.InputField(desc="The user persona to adopt.")
    document_text = dspy.InputField(desc="The full text of a single policy document.")
    question = dspy.OutputField(desc="A realistic user question from the given persona.")
    answer = dspy.OutputField(desc="A detailed, gold-standard answer based only on the document text.")

class GenerateMultiHopQA(dspy.Signature):
    """
    Given the full text of MULTIPLE university policy documents and a user persona, your task is to act as that user and generate one complex, multi-hop question that requires synthesizing information from ALL provided documents to answer. Then, provide a comprehensive, gold-standard answer.

    ---
    **Crucial Guidelines for the Question:**
    - **Focus on a single, plausible user scenario.** The question should be grounded in a real-world problem a person might face.
    - **Be concise and realistic.** The question should sound like something a real person would type or ask. Avoid creating overly academic or convoluted questions that list every possible policy detail.
    - **The goal is REALISM, not maximum logical complexity.**
    ---

    The question must be non-trivial and demonstrate a need to connect concepts across different policies.
    The answer must be detailed, authoritative, and explicitly reference and synthesize facts from **all provided documents**.
    """
    persona = dspy.InputField(desc="The user persona to adopt.")
    documents_context = dspy.InputField(desc="The concatenated full text of multiple distinct policy documents.")
    question = dspy.OutputField(desc="A complex, multi-hop question requiring information from all documents.")
    answer = dspy.OutputField(desc="A detailed, gold-standard answer synthesizing information from all documents.")

class ClassifyPersonaForTopic(dspy.Signature):
    """
    Based on a summary of one or more policy documents, find or create the single most relevant user persona who would be interested in this topic or combination of topics.

    First, review the `persona_options` and select the best fit if one exists.

    If, and only if, none of the provided options accurately represent a realistic user for this specific set of documents, generate a new, more descriptive custom persona. Otherwise, leave the custom persona field empty.
    """
    documents_summary = dspy.InputField(desc="A summary of one or more documents, each with a title and snippet.")
    persona_options: list[str] = dspy.InputField(desc="A list of possible user personas.")
    best_persona = dspy.OutputField(desc="The single best persona. This can be one of the selected options or a new custom persona.")


# --- Generation Modules ---
class PersonaClassifier(dspy.Module):
    """
    A DSPy module that selects or generates a relevant persona for a given document sample. 
    
    This ensures the generated questions are realistic, diverse, and grounded in
    plausible user scenarios, improving the overall quality and coverage of the 
    synthetic dataset.
    """
    def __init__(self):
        super().__init__()
        self.classifier = dspy.Predict(ClassifyPersonaForTopic)

    def forward(self, documents: List[Dict[str, str]]) -> str:
        """
        Takes a list of documents and returns the most appropriate user persona.

        Args:
            documents: A list of document dictionaries, each with 'title' and 'content'.

        Returns:
            The name of the most relevant user persona as a string.
        """
        summaries = [f"Title: {doc['title']}\nDescription: {doc['description']}\nSnippet: {(doc['text'][:300] + '...')}" for doc in documents]
        summary_str = "\n\n---\n\n".join(summaries)
        prediction = self.classifier(documents_summary=summary_str, persona_options=str(USER_PERSONAS))
        persona = prediction.best_persona if prediction.best_persona and len(prediction.best_persona) > 5 else random.choice(USER_PERSONAS)
        if persona not in USER_PERSONAS:
            USER_PERSONAS.append(persona)
            print(f"(New custom persona added: '{persona}')")
        return persona

class SyntheticDataGenerator(dspy.Module):
    """
    A DSPy module that orchestrates the generation of a single synthetic QA pair.
    
    This module uses a PersonaClassifier to determine a relevant user persona,
    then calls either a single-hop or multi-hop generation signature to produce
    a question and answer pair based on the provided document(s).
    
    Attributes:
        teacher_lm: The powerful language model used for the generation task.
    """
    def __init__(self, teacher_lm):
        super().__init__()
        self.persona_classifier = PersonaClassifier()
        self.teacher_lm = teacher_lm
        self.generate_single = dspy.ChainOfThought(GenerateSingleHopQA)
        self.generate_multi = dspy.ChainOfThought(GenerateMultiHopQA)

    def generate_qa_pair(self, documents: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Generates a single question-answer pair from one or more documents.

        Args:
            documents: A list containing one or more document dictionaries.

        Returns:
            A dictionary representing the generated QA example, including the
            question, answer, gold titles, and other metadata.
        """
        persona = self.persona_classifier(documents=documents)
        
        with dspy.context(lm=self.teacher_lm):
            if len(documents) == 1:
                doc = documents[0]
                prediction = self.generate_single(persona=persona, document_text=doc['text'])
            else:
                context = "\n\n--- NEW DOCUMENT ---\n\n".join([doc['text'] for doc in documents])
                prediction = self.generate_multi(persona=persona, documents_context=context)

        return {
            "question": prediction.question,
            "gold_answer": prediction.answer,
            "gold_titles": [d['title'] for d in documents],
            "gold_contents": [d['text'] for d in documents],
            "hop_count": len(documents),
            "persona": persona
        }


# --- Helper Functions ---
def add_distractor_documents(example: Dict[str, Any], all_docs: List[Dict[str, str]], num_distractors: int = 1) -> Dict[str, Any]:
    """
    Adds irrelevant 'distractor' documents to a generated QA example.

    This makes the evaluation more robust by adding noise to the context. It
    tests the synthesis model's ability to ignore irrelevant information and
    strictly adhere to the correct source documents, simulating a retrieval
    step that returned some false positives.

    Args:
        example: The generated QA pair dictionary.
        all_docs: A list of all documents in the corpus to sample from.
        num_distractors: The number of distractor documents to add.

    Returns:
        The example dictionary, now with added 'distractor_titles' and
        'distractor_contents' keys.
    """
    gold_titles = set(example['gold_titles'])
    potential_distractors = [doc for doc in all_docs if doc['title'] not in gold_titles]
    
    num_to_sample = min(num_distractors, len(potential_distractors))
    if num_to_sample > 0:
        distractors = random.sample(potential_distractors, num_to_sample)
        example['distractor_titles'] = [doc['title'] for doc in distractors]
        example['distractor_contents'] = [doc['text'] for doc in distractors]
    else:
        example['distractor_titles'] = []
        example['distractor_contents'] = []
    return example

def save_to_jsonl(data: List[Dict], file_path: Path):
    """
    Saves a list of dictionaries to a JSON Lines (JSONL) file.

    Each dictionary is written as a new line in the file.

    Args:
        data: A list of dictionary objects to save.
        file_path: The path to the output .jsonl file.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    print(f"✅ Saved {len(data)} examples to {file_path}")

def process_and_save_dataset(
    examples: List[Dict],
    all_docs: List[Dict],
    output_dir: Path,
    test_size: int,
    num_distractors: int = 1 # Make num_distractors configurable
) -> None:
    """
    Adds distractors to generated examples, shuffles, splits into train/test sets,
    and saves them as JSONL files.

    Args:
        examples: The list of generated QA example dictionaries.
        all_docs: The full list of documents in the corpus (for sampling distractors).
        output_dir: The directory to save the train.jsonl and test.jsonl files.
        test_size: The desired number of examples for the test set.
        num_distractors: The number of distractor documents to add to each example.
    """
    if not examples:
        logger.error("No examples were provided to process and save. Exiting.")
        return

    logger.info(f"Adding {num_distractors} distractors to {len(examples)} generated examples...")
    final_examples = [
        add_distractor_documents(ex, all_docs, num_distractors)
        for ex in tqdm(examples, desc="Adding Distractors")
    ]
    
    logger.info("Shuffling and splitting dataset...")
    random.shuffle(final_examples)
    output_dir.mkdir(parents=True, exist_ok=True)

    actual_test_size = min(test_size, len(final_examples))
    if actual_test_size == 0 and len(final_examples) > 0:
        logger.warning("Test set size is 0. Saving all examples as training data.")
        test_data = []
        train_data = final_examples
    elif len(final_examples) == 0:
         logger.warning("No final examples to save.")
         test_data = []
         train_data = []
    else:
        test_data = final_examples[:actual_test_size]
        train_data = final_examples[actual_test_size:]

    # Save using the helper function
    logger.info("Saving datasets...")
    save_to_jsonl(train_data, output_dir / "train.jsonl")
    save_to_jsonl(test_data, output_dir / "test.jsonl")
    logger.info(f"Train ({len(train_data)}) and Test ({len(test_data)}) sets saved to {output_dir}")
    