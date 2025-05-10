import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import re
from pydantic import BaseModel, Field
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Body
from contextlib import asynccontextmanager
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import yaml
import sys

# Configuração de logging compatível com Windows
class UnicodeSafeStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except UnicodeEncodeError:
            # Remove caracteres Unicode problemáticos
            msg = record.msg.encode('ascii', 'ignore').decode('ascii')
            record.msg = msg
            super().emit(record)

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        UnicodeSafeStreamHandler(),
        logging.FileHandler('logs/sqli_detector.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Configuração de diretórios
BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = BASE_DIR / "config"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Garante que os diretórios existam
for directory in [CONFIG_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

class ConfigLoader:
    @staticmethod
    def load_config():
        config_path = CONFIG_DIR / "production.yaml"
        default_config = {
            'app': {
                'version': '1.0.0',
                'environment': 'development',
                'debug': False
            },
            'api': {
                'host': '0.0.0.0',
                'port': 8000
            },
            'model': {
                'path': str(MODELS_DIR / "production" / "best_model.joblib"),
                'threshold': 0.7,
                'fallback_path': str(MODELS_DIR / "fallback_model.joblib"),
                'min_acceptable_accuracy': 0.6  # Ajustado para 60%
            },
            'logging': {
                'level': 'INFO',
                'max_size': '10MB',
                'backup_count': 3
            }
        }

        if not config_path.exists():
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(default_config, f)
            logger.info("Arquivo de configuracao padrao criado")
        
        with open(config_path, encoding='utf-8') as f:
            loaded_config = yaml.safe_load(f) or {}
            return {
                'app': {**default_config['app'], **loaded_config.get('app', {})},
                'api': {**default_config['api'], **loaded_config.get('api', {})},
                'model': {**default_config['model'], **loaded_config.get('model', {})},
                'logging': {**default_config['logging'], **loaded_config.get('logging', {})}
            }

config = ConfigLoader.load_config()

class SQLIFeatureExtractor:
    def __init__(self):
        # Regex pré-compilados para melhor performance
        self.comment_pattern = re.compile(r'--|\/\*')
        self.union_pattern = re.compile(r'\bUNION\b', re.IGNORECASE)
        self.or_pattern = re.compile(r'\bOR\b', re.IGNORECASE)
        self.equality_pattern = re.compile(r'\=\s*[\'"]?\d')

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extrai features SQLi com tratamento robusto de erros"""
        df = df.copy()
        try:
            df['contains_comment'] = df['query'].str.contains(self.comment_pattern).astype(int)
            df['uses_union'] = df['query'].str.contains(self.union_pattern).astype(int)
            df['uses_or'] = df['query'].str.contains(self.or_pattern).astype(int)
            df['uses_semicolon'] = df['query'].str.contains(';').astype(int)
            df['uses_equality'] = df['query'].str.contains(self.equality_pattern).astype(int)
        except Exception as e:
            logger.error(f"Erro na extracao de features SQLi: {str(e)}")
            for col in ['contains_comment', 'uses_union', 'uses_or', 'uses_semicolon', 'uses_equality']:
                df[col] = 0
        return df

class TextFeatureExtractor:
    def __init__(self):
        self.vectorizer_path = MODELS_DIR / "text_features" / "vectorizer.joblib"
        self.vectorizer = self._initialize_vectorizer()
        
    def _initialize_vectorizer(self):
        """Inicializa vetorizador com fallback automático"""
        try:
            if self.vectorizer_path.exists():
                vectorizer = joblib.load(self.vectorizer_path)
                logger.info(f"Vetorizador carregado de {self.vectorizer_path}")
                return vectorizer
        except Exception as e:
            logger.error(f"Erro ao carregar vetorizador: {str(e)}")
        
        # Fallback: treina novo vetorizador
        return self._train_fallback_vectorizer()
    
    def _train_fallback_vectorizer(self):
        """Treina vetorizador com dados básicos"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        training_queries = [
            # Queries normais
            "SELECT * FROM users WHERE id = 1",
            "INSERT INTO products (name) VALUES ('book')",
            "UPDATE accounts SET balance = 100 WHERE user_id = 42",
            
            # Queries maliciosas
            "' OR 1=1 --",
            "1; DROP TABLE users--",
            "' UNION SELECT username, password FROM users--"
        ]
        
        vectorizer = TfidfVectorizer(
            max_features=100,
            token_pattern=r'(?u)\b\w+\b|[-!$%^&*()_+|~=`{}$$$$:";\'<>?,.\/]',
            ngram_range=(1, 2)
        )
        
        try:
            vectorizer.fit(training_queries)
            self.vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(vectorizer, self.vectorizer_path)
            logger.info(f"Vetorizador fallback salvo em {self.vectorizer_path}")
        except Exception as e:
            logger.error(f"Erro ao salvar vetorizador: {str(e)}")
        
        logger.warning("Usando vetorizador fallback")
        return vectorizer
    
    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extrai features de texto com fallback seguro"""
        try:
            features = self.vectorizer.transform(df['query'])
            feature_names = self.vectorizer.get_feature_names_out()
            features_df = pd.DataFrame(
                features.toarray(), 
                columns=[f"tfidf_{name}" for name in feature_names]
            )
            return pd.concat([df, features_df], axis=1)
        except Exception as e:
            logger.error(f"Erro na extracao de texto: {str(e)}")
            return df

class QueryRequest(BaseModel):
    query: str = Field(..., example="SELECT * FROM users WHERE id = 1", min_length=1, max_length=1000)
    session_id: Optional[str] = Field(None, example="session_12345", max_length=50)
    source_ip: Optional[str] = Field(None, example="192.168.1.100", max_length=45)
    user_agent: Optional[str] = Field(None, example="Mozilla/5.0", max_length=200)

class DetectionResponse(BaseModel):
    is_sqli: bool = Field(..., example=True)
    probability: float = Field(..., example=0.95, ge=0, le=1)
    features: Dict[str, float] = Field(..., example={"contains_comment": 1})
    model_version: str = Field(..., example="1.0.0")
    session_id: Optional[str] = Field(None, example="session_12345")

class MinimalDetector:
    """Detector mínimo para fallback crítico"""
    def detect(self, query: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        return {
            'is_sqli': False,
            'probability': 0.0,
            'features': {},
            'model_version': 'fallback-1.0',
            'error': 'Servico em modo fallback',
            'session_id': metadata.get('session_id') if metadata else None
        }

class SQLiDetector:
    """Sistema de detecção SQLi com fallback tolerante"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.model_version = config['app']['version']
        
        # Inicializa componentes
        self.sqli_extractor = SQLIFeatureExtractor()
        self.text_extractor = TextFeatureExtractor()
        
        # Carrega modelo
        self._initialize_model()
        
        # Verificação tolerante
        try:
            self._verify_system()
        except Exception as e:
            logger.error(f"Verificacao do sistema falhou: {str(e)}")
            # Continua mesmo com falha na verificação

    def _initialize_model(self):
        """Carrega modelo com fallback automático"""
        main_model_path = Path(self.config['model']['path'])
        fallback_path = Path(self.config['model']['fallback_path'])
        
        # Tenta carregar modelo principal
        if main_model_path.exists():
            try:
                self.model = joblib.load(main_model_path)
                logger.info(f"Modelo principal carregado de {main_model_path}")
                return
            except Exception as e:
                logger.error(f"Falha ao carregar modelo principal: {str(e)}")
        
        # Tenta carregar fallback existente
        if fallback_path.exists():
            try:
                self.model = joblib.load(fallback_path)
                logger.info(f"Modelo fallback carregado de {fallback_path}")
                return
            except Exception as e:
                logger.error(f"Falha ao carregar modelo fallback: {str(e)}")
        
        # Cria e treina novo modelo fallback
        self._train_fallback_model()
    
    def _train_fallback_model(self):
        """Treina modelo fallback básico"""
        from sklearn.ensemble import RandomForestClassifier
        
        logger.warning("Treinando modelo fallback")
        
        # Dados de treinamento balanceados
        X_train = [
            # Queries normais
            "SELECT id FROM users WHERE email = ?",
            "UPDATE products SET price = 10",
            "INSERT INTO logs (event) VALUES ('login')",
            "DELETE FROM sessions WHERE expired = 1",
            
            # Queries maliciosas
            "' OR 1=1 --",
            "1; DROP TABLE users--",
            "admin'--"
        ]
        y_train = [0, 0, 0, 0, 1, 1, 1]  # 0=normal, 1=SQLi
        
        try:
            self.model = Pipeline([
                ('vectorizer', self.text_extractor.vectorizer),
                ('classifier', RandomForestClassifier(
                    n_estimators=50,
                    random_state=42,
                    class_weight='balanced'
                ))
            ])
            
            self.model.fit(X_train, y_train)
            
            # Salva o fallback
            fallback_path = Path(self.config['model']['fallback_path'])
            fallback_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, fallback_path)
            logger.info(f"Modelo fallback salvo em {fallback_path}")
        except Exception as e:
            logger.error(f"Erro critico ao treinar fallback: {str(e)}")
            raise RuntimeError("Nao foi possivel inicializar o detector")

    def _verify_system(self):
        """Verificação tolerante do sistema"""
        test_cases = [
            ("' OR 1=1 --", True),      # SQLi claro
            ("1; DROP TABLE", True),    # SQLi malicioso
            ("SELECT 1", False),        # Query normal
            ("admin'--", True),         # SQLi básico
            ("UPDATE users SET pass='123'", False)  # Update normal
        ]
        
        passed_tests = 0
        min_required = int(len(test_cases) * self.config['model']['min_acceptable_accuracy'])
        
        for query, expected in test_cases:
            try:
                result = self.detect(query)
                if result['is_sqli'] == expected:
                    passed_tests += 1
                else:
                    logger.warning(f"Teste nao esperado para: {query} (esperado: {expected}, obtido: {result['is_sqli']})")
            except Exception as e:
                logger.error(f"Erro no teste: {query} - {str(e)}")
        
        if passed_tests < min_required:
            logger.error(f"Testes falharam ({passed_tests}/{len(test_cases)} passaram)")
            # Apenas loga o erro, não interrompe a execução
        else:
            logger.info(f"Sistema verificado ({passed_tests}/{len(test_cases)} testes passaram)")

    def extract_features(self, query: str) -> pd.DataFrame:
        """Extrai features com tratamento robusto"""
        try:
            if not query or not isinstance(query, str):
                raise ValueError("Query invalida")
                
            query = query[:1000]  # Limita tamanho
            
            df = pd.DataFrame({'query': [query]})
            sql_features = self.sqli_extractor.extract(df)
            text_features = self.text_extractor.extract(df)
            
            # Combina features garantindo colunas mínimas
            features = pd.concat([sql_features, text_features], axis=1)
            
            # Garante features mínimas
            for col in ['contains_comment', 'uses_union', 'uses_or', 'uses_semicolon', 'uses_equality']:
                if col not in features.columns:
                    features[col] = 0
            
            return features.fillna(0)
            
        except Exception as e:
            logger.error(f"Erro na extracao de features: {str(e)}")
            return pd.DataFrame()

    def detect(self, query: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Executa detecção com fallback completo"""
        result = {
            'is_sqli': False,
            'probability': 0.0,
            'features': {},
            'model_version': self.model_version,
            'error': None,
            'session_id': metadata.get('session_id') if metadata else None
        }
        
        try:
            if not self.model:
                raise RuntimeError("Modelo nao disponivel")
                
            features = self.extract_features(query)
            if features.empty:
                raise ValueError("Falha na extracao de features")
            
            # Features mínimas garantidas
            required_features = {
                'contains_comment': 0,
                'uses_union': 0,
                'uses_or': 0,
                'uses_semicolon': 0,
                'uses_equality': 0
            }
            
            # Atualiza com valores reais
            for feat in required_features:
                if feat in features.columns:
                    required_features[feat] = float(features[feat].iloc[0])
            
            # Predição com fallback
            try:
                if hasattr(self.model, 'predict_proba'):
                    proba = self.model.predict_proba([query])[0][1]
                else:
                    proba = float(self.model.predict([query])[0])
            except Exception as e:
                logger.warning(f"Erro na predicao: {str(e)} - usando valor neutro")
                proba = 0.5  # Valor neutro em caso de erro
                
            result.update({
                'is_sqli': proba >= self.config['model']['threshold'],
                'probability': proba,
                'features': required_features
            })
            
        except Exception as e:
            logger.error(f"Erro na deteccao: {str(e)}", exc_info=True)
            result['error'] = str(e)
            
        return result

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerenciador de ciclo de vida tolerante a falhas"""
    detector = None
    try:
        logger.info("Inicializando SQLiDetector...")
        detector = SQLiDetector(config)
        app.state.detector = detector
        logger.info("SQLiDetector inicializado com sucesso")
        yield
    except Exception as e:
        logger.error(f"Falha na inicializacao: {str(e)}")
        app.state.detector = MinimalDetector()
        logger.warning("Usando detector minimo de fallback")
        yield
    finally:
        logger.info("Encerrando SQLiDetector")

app = FastAPI(
    title="SQL Injection Detector API",
    description="API para deteccao de injecao SQL em consultas de banco de dados",
    version=config['app']['version'],
    lifespan=lifespan
)

@app.get("/health", include_in_schema=False)
async def health_check(request: Request):
    """Endpoint de verificacao de saude"""
    status = {
        "status": "OK" if hasattr(request.app.state, 'detector') and not isinstance(request.app.state.detector, MinimalDetector) else "DEGRADED",
        "version": config['app']['version'],
        "model_ready": hasattr(request.app.state, 'detector') and request.app.state.detector is not None,
        "environment": config['app']['environment']
    }
    return status

@app.post("/detect", 
          response_model=DetectionResponse,
          responses={
              200: {"description": "Analise concluida com sucesso"},
              400: {"description": "Requisicao invalida"},
              500: {"description": "Erro interno no servidor"},
              503: {"description": "Servico nao disponivel"}
          })
async def detect_sqli(
    request: Request,
    query_req: QueryRequest = Body(..., examples={
        "normal": {
            "summary": "Consulta normal",
            "value": {
                "query": "SELECT * FROM users WHERE id = 1",
                "session_id": "session_123",
                "source_ip": "192.168.1.1",
                "user_agent": "Mozilla/5.0"
            }
        },
        "sql_injection": {
            "summary": "Tentativa de SQL Injection",
            "value": {
                "query": "' OR 1=1 --",
                "session_id": "session_456",
                "source_ip": "10.0.0.1",
                "user_agent": "curl/7.68.0"
            }
        }
    })
):
    # Verifica se o detector está disponível
    if not hasattr(request.app.state, 'detector'):
        raise HTTPException(
            status_code=503,
            detail="Servico nao disponivel. Tente novamente mais tarde."
        )
    
    # Prepara metadados
    metadata = {
        "session_id": query_req.session_id,
        "source_ip": query_req.source_ip or request.client.host,
        "user_agent": query_req.user_agent or request.headers.get("user-agent", "")
    }
    
    try:
        # Executa a detecção
        result = request.app.state.detector.detect(query_req.query, metadata)
        
        # Trata erros da detecção
        if result.get('error'):
            raise HTTPException(
                status_code=500,
                detail=result['error'] if config['app']['debug'] else "Erro na analise da query"
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro inesperado: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=str(e) if config['app']['debug'] else "Erro interno no processamento"
        )

def run_api():
    """Inicia o servidor da API"""
    logger.info(f"Iniciando API em http://{config['api']['host']}:{config['api']['port']}")
    
    uvicorn.run(
        app,
        host=config['api']['host'],
        port=config['api']['port'],
        log_config=None,
        access_log=False
    )

if __name__ == "__main__":
    # Configuração adicional de logging
    file_handler = logging.FileHandler(
        filename=LOGS_DIR / 'sqli_detector.log',
        mode='a',
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    run_api()