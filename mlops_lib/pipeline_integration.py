from mlops_lib.experiment_tracking import ExperimentTracker
from mlops_lib.model_management import ModelManager
from utils.logger import setup_logger
import pandas as pd

logger = setup_logger(__name__)

class PipelineIntegrator:
    """
    Orchestrates the MLOps pipeline, from training to registration.
    """
    def __init__(self, tracker: ExperimentTracker, manager: ModelManager):
        """
        Initializes the PipelineIntegrator.

        Args:
            tracker (ExperimentTracker): An instance of ExperimentTracker.
            manager (ModelManager): An instance of ModelManager.
        """
        self.tracker = tracker
        self.manager = manager
        logger.info("\n✅ PipelineIntegrator inicializado com sucesso.")

    def run_training_pipeline(
        self,
        trainer,
        evaluator, 
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str,
        params: dict,
        model_flavor: str,
        run_name: str,
        register_threshold: float = None,
        threshold_metric: str = None
    ):
        """
        Executes a complete training and registration pipeline for a model.
        """
        logger.info(f"\n--------------------------------------------------")
        logger.info(f"\n▶️  Iniciando pipeline para o modelo: {model_name}")
        logger.info(f"\n--------------------------------------------------")
        
        with self.tracker.start_run(run_name=run_name) as run:
            # 1. Treinar o modelo
            logger.info("\n🧠  Treinando o modelo...")
            model = trainer.train(X_train, y_train, params)
            logger.info("\n👍  Treinamento concluído.")
            
            # 2. Avaliar o modelo
            logger.info("\n📊  Avaliando o modelo...")
            metrics = evaluator.evaluate(model, X_test, y_test)
            logger.info(f"\n📈  Métricas de avaliação: {metrics}")
            
            # 3. Logar tudo no MLflow
            logger.info("\n📝  Registrando parâmetros e métricas no MLflow...")
            self.tracker.log_params(params)
            self.tracker.log_metrics(metrics)
            
            # 4. Logar o artefato do modelo
            model_path = f"model-{model_flavor}"
            self.tracker.log_model(model, model_flavor, model_path)
            
            # 5. Decidir se o modelo deve ser registrado
            should_register = True
            if register_threshold is not None and threshold_metric is not None:
                metric_value = metrics.get(threshold_metric)
                if metric_value is None or metric_value < register_threshold:
                    should_register = False
                    logger.warning(
                        f"\n⚠️  Modelo não atingiu o limiar de performance. "
                        f"\nMétrica '{threshold_metric}' foi {metric_value:.4f}, "
                        f"\nmas o limiar era {register_threshold}. O modelo não será registrado."
                    )

            if should_register:
                logger.info("\n🏆  Performance excelente! Registrando modelo no Model Registry...")
                model_version = self.manager.register_model(
                    run_id=run.info.run_id,
                    model_path=model_path,
                    model_name=model_name
                )
                
                self.manager.transition_model_stage(
                    model_name=model_name,
                    version=model_version.version,
                    stage="Staging"
                )
            
            logger.info(f"🏁 Pipeline para o modelo {model_name} finalizada.")
