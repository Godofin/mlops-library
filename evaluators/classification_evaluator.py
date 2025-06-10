from evaluators.base_evaluator import BaseEvaluator
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pandas as pd

class ClassificationEvaluator(BaseEvaluator):
    """
    Evaluator for classification models.
    """
    def evaluate(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Evaluates the classification model."""
        preds = model.predict(X_test)
        pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "f1_score": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, pred_proba)
        }
        return metrics
