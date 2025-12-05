import dspy
from dotenv import load_dotenv

import random

from src.rag_system import data_stores, architectures, tools
from src.data_processing.utils import load_dspy_dataset
from src.evaluation.metrics import f1_metric
from src.utils import get_active_config

def main(config_module=None):
    """Main function to compile and save an optimized DSPy agent."""
    # --- 1. Setup Environment ---
    cfg = get_active_config(config_module)

    print("--- Setting up Compilation Environment ---")
    load_dotenv()
    
    # Configure the DSPy LMs
    # A powerful "teacher" model is needed for good compilation results.
    teacher_lm = dspy.LM(cfg.COMPILER_LM_MODEL)
    system_lm = dspy.LM(cfg.RAG_LM_MODEL)
    dspy.configure(lm=system_lm)
    print(f"✅ DSPy LM configured with teacher model: {cfg.COMPILER_LM_MODEL}")

    # --- 2. Load Data and Initialize Base Agent ---
    print("\n--- Loading Data and Initializing Base Agent ---")
    stores = data_stores.PPLDataStores(cfg.as_dict())

    required_fields = ['question', 'gold_titles']
    full_train_set = load_dspy_dataset(cfg.SYNTHETIC_TRAIN_SET_PATH, fields_to_load=required_fields)

    # Split the dataset as required by BootstrapFewShotWithRandomSearch
    random.shuffle(full_train_set) # just in case
    trainset = full_train_set[:50]
    devset = full_train_set[50:]

    # We'll compile the agent that uses agentic chunks, as it's the most powerful.
    agentic_tools = tools.get_final_agentic_tools(
        tools.PPLTools(
            vectorstore=stores.agentic_vectorstore,
            metastore=stores.metadata_vectorstore,
            bm25_retriever=stores.agentic_bm25_retriever,
            query_lm=teacher_lm, # The query expansion can also use the powerful model
            corpus_file_path=cfg.CORPUS_FILE,
        )
    )
    
    # Initialize the un-compiled agent
    uncompiled_agent = architectures.AgenticRAG(tools=agentic_tools)
    uncompiled_retriever = uncompiled_agent.retriever

    # --- 3. Set up the Compiler ---
    print("\n--- Setting up BootstrapFewShot Compiler ---")
    
    # The compiler will try to generate demonstrations that improve the F1 score.
    compiler = dspy.BootstrapFewShotWithRandomSearch(
        metric=f1_metric,
        max_labeled_demos=0,
        teacher_settings=dict(lm=teacher_lm),
        max_errors=99,
        num_candidate_programs=10,
        num_threads=5
    )

    # --- 4. Run Compilation ---
    # This is the long-running step.
    print("\n--- Starting Compilation (This will take a long time!) ---")
    compiled_retriever = compiler.compile(student=uncompiled_retriever, trainset=trainset, valset=devset)

    # --- 5. Save the Optimized Program ---
    print("\n--- Saving Compiled Agent ---")
    save_path = cfg.COMPILED_AGENT_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    compiled_retriever.save(str(save_path))
    print(f"✅ Compiled agent saved successfully to: {save_path}")

if __name__ == "__main__":
    main()