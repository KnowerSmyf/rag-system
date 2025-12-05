import os
import dspy
import pandas as pd
from dotenv import load_dotenv

# Import your project's configuration and modules
from src.rag_system import data_stores, tools
from src.rag_system.architectures import AgenticRAG, VanillaRAG, CrossEncoderReRanker
from src.data_processing.utils import load_dspy_dataset
from src.evaluation import runner
from src.utils import get_active_config

class NonRAG(dspy.Module):
    """A simple baseline that answers questions without any retrieval."""
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        prediction = self.generate_answer(question=question)
        # The output must match the structure of the other RAG systems
        return dspy.Prediction(answer=prediction.answer, titles=[])

def main(config_module=None):
    """Main function to set up and run the entire evaluation pipeline."""
    cfg = get_active_config(config_module)

    # --- 1. Setup Environment ---
    print("--- Setting up Evaluation Environment ---")
    load_dotenv()
    for key in list(os.environ.keys()):
        if "LANGSMITH" in key:
            del os.environ[key]
    
    # Configure DSPy LMs (main system model and external judge model)
    system_lm = dspy.LM(cfg.RAG_LM_MODEL, max_tokens=8000)
    evaluator_lm = dspy.LM(**cfg.EVALUATOR_LM_KWARGS)
    dspy.configure(lm=system_lm, teacher_lm=evaluator_lm) # 'teacher_lm' will be used for SematicF1 metric
    print(f"✅ DSPy LMs cfgured (System: {cfg.RAG_LM_MODEL}, Judge: {cfg.EVALUATOR_LM_MODEL}).")

    # --- 2. Load Data and Systems ---
    print("\n--- Loading Data and Initializing RAG Systems ---")
    stores = data_stores.PPLDataStores(config=cfg.as_dict())
    test_set = load_dspy_dataset(cfg.SYNTHETIC_TEST_SET_PATH)

    # Initialize toolsets for agentic models
    agentic_tools = tools.get_final_agentic_tools(
        tools.PPLTools(
            stores.agentic_vectorstore, 
            stores.metadata_vectorstore, 
            stores.agentic_bm25_retriever, 
            system_lm,
            cfg.CORPUS_FILE
        )
    )
    text_split_tools = tools.get_final_agentic_tools(
        tools.PPLTools(
            stores.textsplit_vectorstore, 
            stores.metadata_vectorstore, 
            stores.textsplit_bm25_retriever, 
            system_lm,
            cfg.CORPUS_FILE
        )
    )

    # Define all system cfgurations to be tested
    systems_to_evaluate = {
        "LLM_Only_Baseline": NonRAG(),
        "VanillaRAG_TextSplit": VanillaRAG(stores.textsplit_vectorstore, CrossEncoderReRanker()),
        "VanillaRAG_Agentic": VanillaRAG(stores.agentic_vectorstore, CrossEncoderReRanker()),
        "AgenticRAG_Uncompiled_TextSplit": AgenticRAG(tools=text_split_tools),
        "AgenticRAG_Uncompiled_Agentic": AgenticRAG(tools=agentic_tools),
        "AgenticRAG_Compiled_TextSplit": AgenticRAG(tools=text_split_tools, retriever_path=cfg.COMPILED_AGENT_PATH),
        "AgenticRAG_Compiled_Agentic": AgenticRAG(tools=agentic_tools, retriever_path=cfg.COMPILED_AGENT_PATH),
    }
    print(f"✅ {len(systems_to_evaluate)} systems initialized for evaluation.")

    # --- 3. Run Evaluation ---
    print("\n--- Starting Evaluation Run (This will take a long time!) ---")
    results_df = runner.run_evaluation(systems_to_evaluate, test_set)
    
    # --- 4. Save Results ---
    print("\n--- Saving Evaluation Results ---")
    detailed_path = cfg.RESULTS_DIR / "final_evaluation_detailed.csv"
    summary_path = cfg.RESULTS_DIR / "final_evaluation_summary.csv"

    results_df.to_csv(detailed_path, index=False)
    print(f"✅ Detailed results saved to {detailed_path}")

    summary_df = results_df.groupby('system_name')[['retrieval_f1', 'semantic_f1']].mean().sort_values(by='semantic_f1', ascending=False)
    summary_df.to_csv(summary_path)
    print(f"✅ Summary results saved to {summary_path}")
    
    print("\n--- Evaluation Complete ---")
    print(summary_df)

if __name__ == "__main__":
    main()