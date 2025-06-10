from evaluators.base_evaluator import BaseEvaluator
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import pandas as pd

class ClusteringEvaluator(BaseEvaluator):
    """
    Evaluator for clustering models.
    """
    def evaluate(self, model, X_test: pd.DataFrame, y_test: pd.Series = None) -> dict:
        """
        Evaluates the clustering model. 
        Note: y_test is ignored for unsupervised models.
        
        Args:
            model: The trained clustering model.
            X_test (pd.DataFrame): The data to evaluate the clusters on.
            y_test (pd.Series, optional): Ignored.

        Returns:
            dict: A dictionary with clustering performance metrics.
        """
        # For clustering, predictions are the cluster labels for each data point
        labels = model.predict(X_test)
        
        # Ensure there's more than one cluster to calculate silhouette score
        if len(set(labels)) > 1:
            silhouette = silhouette_score(X_test, labels)
        else:
            silhouette = -1 # Or 0, or None. -1 indicates invalid clustering for this metric.

        metrics = {
            "silhouette_score": silhouette,
            "calinski_harabasz_score": calinski_harabasz_score(X_test, labels)
        }
        return metrics
