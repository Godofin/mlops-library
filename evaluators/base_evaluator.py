from abc import ABC, abstractmethod
import pandas as pd

class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluators. Defines the common interface.
    """
    @abstractmethod
    def evaluate(self, model, X_test: pd.DataFrame, y_test: pd.Series = None) -> dict:
        """
        Evaluates the trained model.

        Args:
            model: The trained model object.
            X_test (pd.DataFrame): Testing features.
            y_test (pd.Series, optional): Testing target. Can be None for unsupervised models.

        Returns:
            dict: A dictionary of performance metrics.
        """
        pass
