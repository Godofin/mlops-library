from trainers.base_trainer import BaseTrainer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

class RegressionTrainer(BaseTrainer):
    """
    Trainer for regression models.
    """
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, params: dict = None):
        """Trains a Linear Regression model."""
        if params is None:
            params = {}
        model = LinearRegression(**params)
        model.fit(X_train, y_train)
        return model

    def evaluate(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Evaluates the regression model."""
        preds = model.predict(X_test)
        
        metrics = {
            "mse": mean_squared_error(y_test, preds),
            "rmse": np.sqrt(mean_squared_error(y_test, preds)),
            "r2_score": r2_score(y_test, preds)
        }
        return metrics
