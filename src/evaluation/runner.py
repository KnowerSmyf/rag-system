# In src/evaluation/runner.py
import dspy
import pandas as pd
from tqdm import tqdm
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import your metric functions and the refined wrapper
from . import metrics

# --- Helper function for parallel execution ---
def evaluate_single_example(system, system_name, example, metric_definitions):
    """Evaluates a single dspy.Example against a system using multiple metrics."""
    try:
        # 1. Get the prediction from the RAG system
        prediction = system(question=example.question)

        # 2. Calculate all defined metrics using the prediction
        scores = {}
        for name, metric_func in metric_definitions.items():
            try:
                # metric_func is now either f1_metric or GoldAnswerSemanticF1(...)
                score = metric_func(example, prediction)
                scores[name] = score
            except Exception as e:
                print(f"  - WARN: Error calculating metric '{name}' for '{system_name}' on question '{example.question[:50]}...': {e}")
                scores[name] = 0.0

        # 3. Return combined results
        return {
            "system_name": system_name,
            "question": example.question,
            **{f"metric_{k}": v for k, v in scores.items()}, # Store scores
            "predicted_answer": getattr(prediction, 'answer', ''),
            "predicted_titles": getattr(prediction, 'titles', []),
            "gold_titles": example.gold_titles,
            "hop_count": getattr(example, 'hop_count', None),
            "persona": getattr(example, 'persona', None),
            "error": None
        }
    except Exception as e:
        print(f"  - ERROR: System '{system_name}' failed on question '{example.question[:50]}...': {e}")
        return {
            "system_name": system_name,
            "question": example.question,
            "metric_retrieval_f1": 0.0, # Default scores on error
            "metric_semantic_f1": 0.0,
            "error": str(e)
            # Add other fields as None or defaults if needed for DataFrame consistency
        }

def run_evaluation(
    systems_to_evaluate: Dict[str, dspy.Module],
    test_set: List[dspy.Example],
    num_threads: int = 8
):
    """Runs a multi-system, multi-metric evaluation using ThreadPoolExecutor."""

    # Define metrics ONCE, including the refined wrapper
    metric_definitions = {
        "retrieval_f1": metrics.f1_metric,
        "semantic_f1": metrics.GoldAnswerSemanticF1(decompositional=True)
    }

    all_results = []

    print(f"\n--- Starting Multi-Threaded Evaluation (max_workers={num_threads}) ---")
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        # Submit all system/example pairs
        for name, system in systems_to_evaluate.items():
            for example in test_set:
                futures.append(executor.submit(
                    evaluate_single_example,
                    system, name, example, metric_definitions
                ))

        # Collect results as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating Systems"):
            result = future.result()
            if result:
                all_results.append(result)

    # Clean up column names and ensure columns exist
    final_df = pd.DataFrame(all_results)
    final_df.columns = [col.replace('metric_', '') if col.startswith('metric_') else col for col in final_df.columns]
    if 'retrieval_f1' not in final_df.columns: final_df['retrieval_f1'] = 0.0
    if 'semantic_f1' not in final_df.columns: final_df['semantic_f1'] = 0.0
    final_df['retrieval_f1'] = final_df['retrieval_f1'].fillna(0.0)
    final_df['semantic_f1'] = final_df['semantic_f1'].fillna(0.0)

    return final_df