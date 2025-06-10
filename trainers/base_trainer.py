from abc import ABC, abstractmethod
import pandas as pd

class BaseTrainer(ABC):
    """
    Abstract base class for all trainers. Defines the common interface.
    The sole responsibility of a trainer is to train a model.
    """
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, params: dict = None):
        """
        Trains a model.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            params (dict, optional): Hyperparameters for the model.

        Returns:
            The trained model object.
        """
        pass
