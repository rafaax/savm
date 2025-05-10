import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from models.model_manager import ModelManager
from features.sqli_features import SQLIFeatureExtractor
from features.text_features import TextFeatureExtractor

# Configuração básica de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SQLiDetector:
    """
    Classe principal para detecção de SQL Injection.
    Combina feature extraction com modelos treinados.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa o detector com configuração personalizável.
        
        Args:
            config: Dicionário com:
                - model_path: Caminho para o modelo salvo
                - threshold: Limiar de decisão (0-1)
        """
        self.config = config or {}
        self._load_components()
        
    def _load_components(self):
        """Carrega todos os componentes necessários."""
        # Feature Extractors
        self.sqli_extractor = SQLIFeatureExtractor()
        self.text_extractor = TextFeatureExtractor()
        
        # Model Manager
        self.model_manager = ModelManager(
            config=self.config.get('model_config', {}),
            models_dir=self.config.get('models_dir', 'models')
        )
        
        # Carrega o modelo
        model_path = self.config.get('model_path')
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Carrega um modelo treinado."""
        try:
            self.model = self.model_manager.load_model(model_path)
            logger.info(f"Modelo carregado de {model_path}")
        except Exception as e:
            logger.error(f"Falha ao carregar modelo: {str(e)}")
            raise
    
    def extract_features(self, query: str) -> pd.DataFrame:
        """
        Extrai features de uma query SQL.
        
        Args:
            query: String com a consulta SQL
            
        Returns:
            DataFrame com features extraídas
        """
        df = pd.DataFrame({'query': [query]})
        
        # Extração de features
        df = self.sqli_extractor.extract(df)
        df = self.text_extractor.extract(df, fit_models=False)
        
        return df
    
    def detect(self, query: str) -> Dict[str, Any]:
        """
        Analisa uma query e retorna resultados da detecção.
        
        Args:
            query: Consulta SQL a ser analisada
            
        Returns:
            Dicionário com:
                - is_sqli: Bool (True se for SQLi)
                - probability: Probabilidade (0-1)
                - features: Features relevantes
        """
        try:
            if not hasattr(self, 'model'):
                raise ValueError("Modelo não carregado")
            
            # Extrai features
            features = self.extract_features(query)
            
            # Faz a predição
            proba = self.model['model'].predict_proba(features)[0][1]
            is_sqli = proba >= self.config.get('threshold', 0.5)
            
            # Features mais importantes
            top_features = self._get_top_features(features)
            
            return {
                'is_sqli': bool(is_sqli),
                'probability': float(proba),
                'features': top_features,
                'feature_vector': features.iloc[0].to_dict()
            }
            
        except Exception as e:
            logger.error(f"Erro na detecção: {str(e)}")
            raise
    
    def _get_top_features(self, features: pd.DataFrame) -> Dict[str, float]:
        """Identifica as features mais relevantes para a decisão."""
        if not hasattr(self.model['model'], 'feature_importances_'):
            return {}
            
        importance = self.model['model'].feature_importances_
        top_indices = importance.argsort()[-5:][::-1]
        
        return {
            features.columns[i]: float(importance[i])
            for i in top_indices
        }

# API Setup (Opcional)
app = FastAPI(title="SQL Injection Detector API")

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

@app.post("/detect")
async def detect_sqli(query_req: QueryRequest):
    """Endpoint para detecção via API."""
    try:
        detector = app.state.detector
        result = detector.detect(query_req.query)
        return {
            **result,
            "session_id": query_req.session_id,
            "model_version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def initialize_api(detector: SQLiDetector):
    """Inicializa a API com uma instância do detector."""
    app.state.detector = detector
    return app

# Exemplo de uso direto
if __name__ == "__main__":
    # Configuração
    config = {
        "model_path": "models/production/best_model.joblib",
        "threshold": 0.7,
        "model_config": {
            "sampling": "smote",
            "random_state": 42
        }
    }
    
    # Inicialização
    detector = SQLiDetector(config)
    
    # Teste
    test_queries = [
        "SELECT * FROM users WHERE id = 1",
        "' OR '1'='1' --",
        "DROP TABLE users"
    ]
    
    for query in test_queries:
        result = detector.detect(query)
        print(f"Query: {query}")
        print(f"Resultado: {'SQLi' if result['is_sqli'] else 'Safe'}")
        print(f"Probabilidade: {result['probability']:.4f}")
        print("---")