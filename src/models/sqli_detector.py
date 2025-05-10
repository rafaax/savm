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
        """
        Inicializa o detector com configuração.
        """
        self.config = config or {
            'model_config': {},
            'models_dir': 'models'
        }
        
        # Inicializa os componentes na ordem correta
        self.sqli_extractor = SQLIFeatureExtractor(
            **self.config.get('sqli_features', {})
        )
        self.text_extractor = TextFeatureExtractor(
            **self.config.get('text_features', {})
        )
        self.model_manager = ModelManager(
            config=self.config.get('model_config', {}),
            models_dir=self.config.get('models_dir', 'models')
        )
        
        # Treina o vetorizador
        self._train_text_vectorizer()
        
        # Carrega o modelo principal
        model_path = self.config.get('model_path')
        if model_path:
            self.load_model(model_path)
        else:
            logger.warning("Nenhum caminho de modelo especificado")
            self._initialize_fallback()

    def _train_text_vectorizer(self):
        """Treina o vetorizador com exemplos iniciais balanceados"""
        training_queries = [
            # Consultas normais
            "SELECT * FROM users",
            "INSERT INTO products VALUES (1, 'book')",
            "UPDATE customers SET name = 'John' WHERE id = 1",
            "DELETE FROM logs WHERE date < '2023-01-01'",
            
            # Consultas maliciosas
            "admin' OR 1=1 --",
            "' UNION SELECT passwords FROM users --",
            "1; DROP TABLE users--",
            "SELECT * FROM information_schema.tables"
        ]
        
        df_train = pd.DataFrame({
            'query': training_queries,
            'label': [0, 0, 0, 0, 1, 1, 1, 1]  # 0=normal, 1=malicioso
        })
        
        self.text_extractor.extract(df_train, fit_models=True)
    
    def _initialize_fallback(self):
        """Inicializa um modelo dummy como fallback"""
        from sklearn.ensemble import RandomForestClassifier
        self.model = {
            'model': RandomForestClassifier(n_estimators=10),
            'metadata': {'fallback': True}
        }
        
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
        """Extrai features de forma segura"""
        try:
            df = pd.DataFrame({'query': [query]})
            
            # Extração segura
            df = self.sqli_extractor.extract(df)
            text_features = self.text_extractor.extract(df, fit_models=False)
            
            # Garante compatibilidade de features
            missing_cols = set(df.columns) - set(text_features.columns)
            for col in missing_cols:
                text_features[col] = df[col]
                
            return text_features.fillna(0)  # Substitui NaN por 0
            
        except Exception as e:
            logger.error(f"Erro seguro na extração: {str(e)}")
            # Retorna dataframe vazio em caso de erro
            return pd.DataFrame()
    
    def detect(self, query: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Versão com tratamento robusto de erros"""
        result = {
            'is_sqli': False,
            'probability': 0.0,
            'features': {},
            'metadata': metadata or {},
            'model_version': self.config.get('model_version', '1.0.0'),
            'error': None
        }
        
        try:
            if not query or not isinstance(query, str):
                raise ValueError("Query inválida ou vazia")
                
            if not hasattr(self, 'model'):
                raise RuntimeError("Modelo não carregado")
            
            # Extração de features com fallback
            try:
                features = self.extract_features(query)
                if features.empty:
                    raise ValueError("Falha na extração de features")
            except Exception as e:
                logger.warning(f"Falha na extração: {str(e)}")
                features = pd.DataFrame({'query': [query]})
                
            # Predição segura
            try:
                proba = self.model['model'].predict_proba(features)[0][1]
                result.update({
                    'is_sqli': proba >= self.config.get('threshold', 0.5),
                    'probability': proba
                })
            except Exception as e:
                logger.error(f"Falha na predição: {str(e)}")
                result['error'] = "Falha na análise"
                
        except Exception as e:
            logger.critical(f"Erro crítico: {str(e)}", exc_info=True)
            result['error'] = str(e)
            
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
        
        app.state.detector = SQLiDetector({
            "model_path": model_path,
            "threshold": 0.7,
            "log_queries": True,
            "model_config": {
                "cache_dir": "cache",
                "preload": True
            },
            "models_dir": "models"
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
        logger.info(f"Iniciando análise para query: {query_req.query[:50]}...")
        
        detector = request.app.state.detector
        
        metadata = {
            "source_ip": query_req.source_ip or request.client.host,
            "user_agent": query_req.user_agent or request.headers.get("user-agent")
        }
        
        logger.debug("Extraindo features...")
        result = detector.detect(query_req.query, metadata)
        logger.info(f"Análise concluída - SQLi: {result['is_sqli']} (Prob: {result['probability']:.2f})")
        
        return {
            "is_sqli": result["is_sqli"],
            "probability": result["probability"],
            "features": result["features"],
            "model_version": result["model_version"],
            "session_id": query_req.session_id
        }
        
    except Exception as e:
        logger.error("Erro detalhado:", exc_info=True)  # Isso logará o traceback completo
        raise HTTPException(
            status_code=500,
            detail=f"Erro interno: {str(e)}" if config.get("debug") else "Erro interno na análise"
        )

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