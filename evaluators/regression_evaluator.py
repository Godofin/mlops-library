from evaluators.base_evaluator import BaseEvaluator
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

class RegressionEvaluator(BaseEvaluator):
    """
    Evaluator for regression models.
    """
    def evaluate(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Evaluates the regression model."""
        preds = model.predict(X_test)
        
        metrics = {
            "mse": mean_squared_error(y_test, preds),
            "rmse": np.sqrt(mean_squared_error(y_test, preds)),
            "r2_score": r2_score(y_test, preds)
        }
        return metrics
