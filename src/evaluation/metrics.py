"""Houses all evaluation metric functions and DSPy signatures for assessment."""
import dspy
import copy
from typing import Dict, Callable
from dspy.evaluate import SemanticF1

def f1_metric(gold: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """Calculates the F1 score for retrieved titles."""
    gold_titles = set(gold.gold_titles)
    predicted_titles_list = getattr(pred, 'titles', None) or []
    pred_titles = set(predicted_titles_list)

    if not gold_titles: return 1.0 if not pred_titles else 0.0
    if not pred_titles: return 0.0

    precision = len(gold_titles.intersection(pred_titles)) / len(pred_titles)
    recall = len(gold_titles.intersection(pred_titles)) / len(gold_titles)
    
    if (precision + recall) == 0: return 0.0
    return (2 * precision * recall) / (precision + recall)


class GoldAnswerSemanticF1(dspy.Module):
    """
    A wrapper for dspy.evaluate.SemanticF1 that correctly uses a 'teacher_lm',
    adapts the 'gold_answer' field, and ensures the prediction has a 'response' field.
    """
    def __init__(self, **kwargs):
        super().__init__()
        # Pass kwargs (like decompositional=True) to the underlying metric
        self.metric = SemanticF1(**kwargs)

    def forward(self, example, pred, trace=None):
        teacher_lm = dspy.settings.get('teacher_lm')
        if not teacher_lm:
            raise ValueError("teacher_lm not configured in dspy.settings...")
        
        with dspy.context(lm=teacher_lm):
            # Adapt the gold example
            temp_example = copy.copy(example)
            if hasattr(temp_example, 'gold_answer'):
                temp_example.response = temp_example.gold_answer
            
            # Create a new dspy.Prediction object mapping the original 'answer' 
            # to the expected 'response'.
            metric_pred = dspy.Prediction(
                response=getattr(pred, 'answer', '') # Use getattr for safety
            )
            return self.metric(temp_example, metric_pred, trace=trace)


class MultiMetricEvaluator(dspy.Module):
    """A DSPy module to calculate multiple evaluation metrics at once."""
    def __init__(self, metrics: Dict[str, Callable]):
        super().__init__()
        self.metrics = metrics

    def forward(self, example, pred, trace=None):
        scores = {}
        for name, metric_func in self.metrics.items():
            try:
                score = metric_func(example, pred, trace=trace)
                scores[name] = score
            except Exception as e:
                print(f"  - WARN: Error calculating metric '{name}' for question '{example.question[:50]}...': {e}")
                scores[name] = 0.0
        # Return a Prediction object containing all scores
        return dspy.Prediction(**scores)