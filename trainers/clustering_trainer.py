from trainers.base_trainer import BaseTrainer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import pandas as pd

class ClusteringTrainer(BaseTrainer):
    """
    Trainer for clustering models. Note: y_train/y_test are not used.
    """
    def train(self, X_train: pd.DataFrame, y_train: pd.Series = None, params: dict = None):
        """Trains a K-Means model."""
        if params is None:
            params = {}
        model = KMeans(**params, random_state=42)
        model.fit(X_train)
        return model

    def evaluate(self, model, X_test: pd.DataFrame, y_test: pd.Series = None) -> dict:
        """Evaluates the clustering model."""
        labels = model.predict(X_test)
        
        metrics = {
            "silhouette_score": silhouette_score(X_test, labels),
            "calinski_harabasz_score": calinski_harabasz_score(X_test, labels)
        }
        return metrics
