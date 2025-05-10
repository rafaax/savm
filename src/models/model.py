import logging
import joblib
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from imblearn.pipeline import make_pipeline as make_imb_pipeline
from imblearn.over_sampling import SMOTE

class SQLiDetectionModel(BaseEstimator, ClassifierMixin):
    """
    Modelo especializado para detecção de SQL Injection com suporte a múltiplos algoritmos.
    
    Args:
        model_type: Tipo de modelo ('random_forest', 'xgboost', 'svm', 'gradient_boosting')
        model_params: Parâmetros específicos do modelo
        balance_method: Método para balanceamento de classes ('smote', 'undersample', None)
    """
    
    MODEL_TYPES = {
        'random_forest': RandomForestClassifier,
        'xgboost': XGBClassifier,
        'svm': SVC,
        'gradient_boosting': GradientBoostingClassifier
    }
    
    def __init__(self, 
                 model_type: str = 'random_forest',
                 model_params: Optional[Dict[str, Any]] = None,
                 balance_method: Optional[str] = 'smote'):
        self.model_type = model_type
        self.model_params = model_params or {}
        self.balance_method = balance_method
        self.logger = logging.getLogger(__name__)
        self._init_model()
    
    def _init_model(self):
        """Inicializa o pipeline do modelo."""
        if self.model_type not in self.MODEL_TYPES:
            raise ValueError(f"Model type must be one of {list(self.MODEL_TYPES.keys())}")
        
        # Modelo base
        model_class = self.MODEL_TYPES[self.model_type]
        self.model = model_class(**self.model_params)
        
        # Pipeline com balanceamento
        if self.balance_method == 'smote':
            self.pipeline = make_imb_pipeline(
                SMOTE(random_state=42),
                self.model
            )
        else:
            self.pipeline = Pipeline([('model', self.model)])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'SQLiDetectionModel':
        """
        Treina o modelo com os dados fornecidos.
        
        Args:
            X: DataFrame com features
            y: Série com labels
            
        Returns:
            self
        """
        try:
            self.logger.info(f"Training {self.model_type} model...")
            self.pipeline.fit(X, y)
            self.logger.info("Model trained successfully")
            return self
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Faz predições para novos dados.
        
        Args:
            X: DataFrame com features
            
        Returns:
            Série com predições (0: normal, 1: SQLi)
        """
        try:
            return self.pipeline.predict(X)
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            raise

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Retorna probabilidades para cada classe.
        
        Args:
            X: DataFrame com features
            
        Returns:
            DataFrame com probabilidades [P(0), P(1)]
        """
        try:
            if hasattr(self.pipeline, 'predict_proba'):
                return self.pipeline.predict_proba(X)
            raise NotImplementedError("This model doesn't support probability predictions")
        except Exception as e:
            self.logger.error(f"Error during probability prediction: {str(e)}")
            raise

    def cross_validate(self, 
                     X: pd.DataFrame, 
                     y: pd.Series,
                     cv: int = 5,
                     metrics: List[str] = ['accuracy', 'f1', 'roc_auc']) -> Dict[str, Any]:
        """
        Executa validação cruzada.
        
        Args:
            X: Features
            y: Labels
            cv: Número de folds
            metrics: Métricas a serem calculadas
            
        Returns:
            Dicionário com resultados da validação
        """
        try:
            scores = cross_validate(
                self.pipeline,
                X,
                y,
                cv=cv,
                scoring=metrics,
                return_train_score=True
            )
            return scores
        except Exception as e:
            self.logger.error(f"Error during cross-validation: {str(e)}")
            raise

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Salva o modelo em disco.
        
        Args:
            filepath: Caminho para salvar o modelo
        """
        try:
            joblib.dump(self, filepath)
            self.logger.info(f"Model saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'SQLiDetectionModel':
        """
        Carrega um modelo salvo.
        
        Args:
            filepath: Caminho para o modelo salvo
            
        Returns:
            Instância do modelo carregado
        """
        try:
            return joblib.load(filepath)
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise

    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Retorna a importância das features (se disponível).
        
        Returns:
            Série com importância das features ou None
        """
        try:
            if hasattr(self.model, 'feature_importances_'):
                return pd.Series(
                    self.model.feature_importances_,
                    index=self.feature_names
                )
            return None
        except Exception as e:
            self.logger.warning(f"Could not get feature importance: {str(e)}")
            return None

    @property
    def feature_names(self) -> Optional[List[str]]:
        """Retorna os nomes das features se disponível."""
        try:
            if hasattr(self.pipeline, 'feature_names_in_'):
                return list(self.pipeline.feature_names_in_)
            return None
        except Exception as e:
            self.logger.warning(f"Could not get feature names: {str(e)}")
            return None