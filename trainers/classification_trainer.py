from trainers.base_trainer import BaseTrainer
from sklearn.linear_model import LogisticRegression
import pandas as pd

class ClassificationTrainer(BaseTrainer):
    """
    Trainer for classification models.
    Its only responsibility is to fit the model. Evaluation is handled by Evaluator classes.
    """
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, params: dict = None):
        """Trains a Logistic Regression model."""
        if params is None:
            params = {}
        model = LogisticRegression(**params, random_state=42)
        model.fit(X_train, y_train)
        return model

# O método evaluate() foi removido. Os outros trainers (Regression, Clustering, etc.)
# seguem o mesmo padrão de terem apenas o método train().
