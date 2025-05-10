import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager
from models.model_manager import ModelManager
from features.sqli_features import SQLIFeatureExtractor
from features.text_features import TextFeatureExtractor
from utils.database import database
from utils.config import config
from utils.logger import logger

class SQLiDetector:
    """Sistema completo de detecção de SQL Injection com API integrada."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        print(f"🔍 Procurando modelo em: {self.config.get('model_path')}")
        try:
            self._initialize_components()
        except Exception as e:
            logger.error(f"Erro na inicialização: {str(e)}")
            self._initialize_fallback()
    
    def _initialize_fallback(self):
        """Inicializa um modelo dummy como fallback"""
        from sklearn.ensemble import RandomForestClassifier
        self.model = {
            'model': RandomForestClassifier(n_estimators=10),
            'metadata': {'fallback': True}
        }
        logger.warning("⚠️ Usando modelo dummy - apenas para desenvolvimento!")
        
    def _initialize_components(self):
        """Inicializa todos os componentes do sistema."""
        logger.info("Inicializando componentes do SQLiDetector")
        
        self.sqli_extractor = SQLIFeatureExtractor()
        self.text_extractor = TextFeatureExtractor()
        self.model_manager = ModelManager()
        
        model_path = self.config.get('model_path')
        if model_path:
            self.load_model(model_path)
        else:
            logger.warning("Nenhum caminho de modelo especificado")
            self._initialize_fallback()
    
    def load_model(self, model_path: str) -> None:
        """Carrega um modelo treinado do disco."""
        try:
            logger.info(f"Carregando modelo de {model_path}")
            self.model = self.model_manager.load_model(model_path)
            logger.info("✅ Modelo carregado com sucesso")
        except Exception as e:
            logger.error(f"❌ Falha ao carregar modelo: {str(e)}")
            raise RuntimeError(f"Erro ao carregar modelo: {str(e)}")

    def extract_features(self, query: str) -> pd.DataFrame:
        """Extrai features de uma query SQL."""
        try:
            df = pd.DataFrame({'query': [query]})
            df = self.sqli_extractor.extract(df)
            df = self.text_extractor.extract(df, fit_models=False)
            return df
        except Exception as e:
            logger.error(f"Erro na extração de features: {str(e)}")
            raise
    
    def detect(self, query: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Analisa uma query SQL e retorna resultados da detecção."""
        result = {
            'is_sqli': False,
            'probability': 0.0,
            'features': {},
            'metadata': metadata or {},
            'model_version': self.config.get('model_version', '1.0.0')
        }
        
        try:
            if not hasattr(self, 'model'):
                raise RuntimeError("Nenhum modelo carregado")
            
            features = self.extract_features(query)
            proba = self.model['model'].predict_proba(features)[0][1]
            is_sqli = proba >= self.config.get('threshold', 0.5)
            
            result.update({
                'is_sqli': is_sqli,
                'probability': proba,
                'features': self._get_top_features(features),
                'feature_vector': features.iloc[0].to_dict()
            })
            
            if self.config.get('log_queries', True):
                self._log_detection(query, result, metadata)
                
        except Exception as e:
            logger.error(f"Erro na detecção: {str(e)}")
            raise
        
        return result
    
    def _get_top_features(self, features: pd.DataFrame) -> Dict[str, float]:
        """Identifica as features mais relevantes."""
        if not hasattr(self.model['model'], 'feature_importances_'):
            return {}
            
        try:
            importance = self.model['model'].feature_importances_
            top_indices = importance.argsort()[-5:][::-1]
            return {
                features.columns[i]: float(importance[i])
                for i in top_indices
            }
        except Exception as e:
            logger.warning(f"Erro ao obter feature importance: {str(e)}")
            return {}
    
    def _log_detection(self, query: str, result: Dict, metadata: Optional[Dict]) -> None:
        """Registra a detecção no banco de dados."""
        try:
            log_data = {
                'query': query,
                'is_sqli': result['is_sqli'],
                'probability': result['probability'],
                'source_ip': metadata.get('source_ip') if metadata else None,
                'user_agent': metadata.get('user_agent') if metadata else None
            }
            database.log_query(log_data)
        except Exception as e:
            logger.error(f"Erro ao registrar query no banco: {str(e)}")

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None

class DetectionResponse(BaseModel):
    is_sqli: bool
    probability: float
    features: Dict[str, float]
    model_version: str
    session_id: Optional[str]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerenciador de ciclo de vida da API"""
    try:
        model_path = "models/production/best_model.joblib"
        if not Path(model_path).exists():
            logger.warning(f"Arquivo de modelo não encontrado: {model_path}")
        
        app.state.detector = SQLiDetector({
            "model_path": model_path,
            "threshold": 0.7,
            "log_queries": True
        })
        yield
    except Exception as e:
        logger.error(f"Falha crítica na inicialização: {str(e)}")
        raise

app = FastAPI(
    title="SQL Injection Detector API",
    description="API para detecção de SQL Injection",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/detect", response_model=DetectionResponse)
async def detect_sqli(request: Request, query_req: QueryRequest):
    try:
        detector = request.app.state.detector
        metadata = {
            "source_ip": query_req.source_ip or request.client.host,
            "user_agent": query_req.user_agent or request.headers.get("user-agent")
        }
        
        result = detector.detect(query_req.query, metadata)
        
        return {
            "is_sqli": result["is_sqli"],
            "probability": result["probability"],
            "features": result["features"],
            "model_version": result["model_version"],
            "session_id": query_req.session_id
        }
    except Exception as e:
        logger.error(f"Erro na API: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro interno na análise")

def run_api(host: str = "0.0.0.0", port: int = 8000):
    """Inicia o servidor da API."""
    logger.info(f"🚀 Iniciando API em http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    try:
        config.load_from_file("config/production.yaml")
    except FileNotFoundError:
        logger.warning("⚠️ Arquivo de configuração não encontrado, usando padrões")
        config.update({
            "api": {"host": "0.0.0.0", "port": 8000},
            "model": {"path": "models/production/best_model.joblib"}
        })
    
    run_api(
        host=config.get("api.host", "0.0.0.0"),
        port=config.get("api.port", 8000)
    )