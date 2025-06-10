import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression, make_blobs

# Componentes da biblioteca MLOps
from utils.config_loader import ConfigLoader
from utils.logger import setup_logger
from mlops_lib.experiment_tracking import ExperimentTracker
from mlops_lib.model_management import ModelManager
from mlops_lib.pipeline_integration import PipelineIntegrator

# Trainers
from trainers.classification_trainer import ClassificationTrainer
from trainers.regression_trainer import RegressionTrainer
from trainers.clustering_trainer import ClusteringTrainer

# --- NOVOS IMPORTS: EVALUATORS ---
from evaluators.classification_evaluator import ClassificationEvaluator
from evaluators.regression_evaluator import RegressionEvaluator
from evaluators.clustering_evaluator import ClusteringEvaluator


def main():
    """
    Função principal para demonstrar o uso da biblioteca MLOps.
    """
    logger = setup_logger("main_script")
    logger.info("\n######################################################")
    logger.info("\n#####     INICIANDO DEMO DA BIBLIOTECA MLOPS     #####")
    logger.info("\n######################################################")

    # 1. Carregar Configurações
    config_loader = ConfigLoader(config_path="config/config.yaml")
    config = config_loader.load()
    mlflow_config = config['mlflow_config']
    params_config = config['model_params']

    # 2. Inicializar os componentes da biblioteca MLOps
    tracker = ExperimentTracker(
        tracking_uri=mlflow_config['tracking_uri'],
        experiment_name=mlflow_config['experiment_name']
    )
    manager = ModelManager(tracking_uri=mlflow_config['tracking_uri'])
    pipeline_integrator = PipelineIntegrator(tracker, manager)

    # --- INSTANCIAR EVALUATORS ---
    classification_evaluator = ClassificationEvaluator()
    regression_evaluator = RegressionEvaluator()
    clustering_evaluator = ClusteringEvaluator()


    # 3. Executar pipeline de classificação
    X_class, y_class = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
    pipeline_integrator.run_training_pipeline(
        trainer=ClassificationTrainer(),
        evaluator=classification_evaluator, # Passando o evaluator
        X_train=pd.DataFrame(X_train), y_train=pd.Series(y_train),
        X_test=pd.DataFrame(X_test), y_test=pd.Series(y_test),
        model_name="CreditRiskClassifier",
        params=params_config['classification'],
        model_flavor="sklearn",
        run_name="logistic_regression_run",
        register_threshold=0.9,
        threshold_metric="roc_auc"
    )

    # 4. Executar pipeline de regressão
    X_reg, y_reg = make_regression(n_samples=1000, n_features=10, noise=25, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    pipeline_integrator.run_training_pipeline(
        trainer=RegressionTrainer(),
        evaluator=regression_evaluator, # Passando o evaluator
        X_train=pd.DataFrame(X_train), y_train=pd.Series(y_train),
        X_test=pd.DataFrame(X_test), y_test=pd.Series(y_test),
        model_name="HousePriceRegressor",
        params=params_config['regression'],
        model_flavor="sklearn",
        run_name="linear_regression_run",
        register_threshold=0.75,
        threshold_metric="r2_score"
    )

    # 5. Executar pipeline de clusterização
    X_clust, _ = make_blobs(n_samples=1000, centers=3, n_features=10, random_state=42)
    pipeline_integrator.run_training_pipeline(
        trainer=ClusteringTrainer(),
        evaluator=clustering_evaluator, # Passando o evaluator
        X_train=pd.DataFrame(X_clust), y_train=None,
        X_test=pd.DataFrame(X_clust), y_test=None,
        model_name="CustomerSegmentation",
        params=params_config['clustering'],
        model_flavor="sklearn",
        run_name="kmeans_clustering_run",
        register_threshold=0.5,
        threshold_metric="silhouette_score"
    )
    
    logger.info("######################################################")
    logger.info("#####            DEMONSTRAÇÃO FINALIZADA           #####")
    logger.info("######################################################")


if __name__ == "__main__":
    main()
