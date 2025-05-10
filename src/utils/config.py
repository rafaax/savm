import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from dotenv import load_dotenv

class ConfigManager:
    """
    Gerenciador centralizado de configurações para o sistema de detecção de SQLi.
    
    Suporta múltiplos formatos (JSON, YAML, .env) com fallback hierárquico.
    """
    
    def __init__(self):
        self._config: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
        self._load_env_vars()
        self._load_defaults()
    
    def _load_env_vars(self) -> None:
        """Carrega variáveis de ambiente do arquivo .env."""
        load_dotenv()
        
        # Mapeamento de env vars para configurações
        env_mappings = {
            'SQLI_MODEL_PATH': ('model.path', str),
            'SQLI_THRESHOLD': ('detection.threshold', float),
            'SQLI_LOGGING_LEVEL': ('logging.level', str),
            'DB_CONN_STRING': ('database.connection_string', str)
        }
        
        for env_var, (config_key, type_cast) in env_mappings.items():
            if env_var in os.environ:
                try:
                    self._config[config_key] = type_cast(os.environ[env_var])
                except ValueError as e:
                    self.logger.warning(f"Failed to parse {env_var}: {str(e)}")

    def _load_defaults(self) -> None:
        """Configurações padrão fallback."""
        default_config = {
            'model': {
                'path': 'models/production/best_model.joblib',
                'type': 'random_forest',
                'threshold': 0.7,
                'retrain_frequency': 'weekly'
            },
            'features': {
                'sqli': {
                    'min_keyword_length': 3,
                    'max_query_length': 10000
                },
                'text': {
                    'max_features': 100,
                    'svd_components': 20
                }
            },
            'logging': {
                'file': 'logs/sqli_detector.log',
                'max_size': 10 * 1024 * 1024,  # 10MB
                'backup_count': 5,
                'level': 'INFO',
                'email': {
                    'enabled': False,
                    'host': 'smtp.example.com',
                    'port': 587,
                    'from': 'alerts@sqli-detector.com',
                    'to': ['devops@example.com'],
                    'username': 'user',
                    'password': 'pass'
                }
            },
            'api': {
                'enabled': True,
                'host': '0.0.0.0',
                'port': 8000,
                'rate_limit': '100/minute'
            },
            'monitoring': {
                'enabled': False,
                'prometheus_port': 9090
            },
            'database': {
                'path': 'data/sqli_detector.db',
                'log_queries': True,
                'log_models': True
            }
        }
        
        # Merge com configurações existentes
        self._merge_configs(default_config)

    def load_from_file(self, file_path: str) -> None:
        """
        Carrega configurações de um arquivo (JSON ou YAML).
        
        Args:
            file_path: Caminho para o arquivo de configuração
        """
        path = Path(file_path)
        if not path.exists():
            self.logger.warning(f"Arquivo de configuração não encontrado: {file_path}")
            return
            
        try:
            with open(path, 'r') as f:
                if path.suffix == '.json':
                    new_config = json.load(f)
                elif path.suffix in ('.yaml', '.yml'):
                    new_config = yaml.safe_load(f)
                else:
                    self.logger.error(f"Formato não suportado: {path.suffix}")
                    return
                
                self._merge_configs(new_config)
                self.logger.info(f"Configurações carregadas de {file_path}")
                
        except Exception as e:
            self.logger.error(f"Erro ao carregar {file_path}: {str(e)}")

    def _merge_configs(self, new_config: Dict[str, Any]) -> None:
        """Combina dicionários de configuração de forma recursiva."""
        for key, value in new_config.items():
            if isinstance(value, dict) and key in self._config:
                self._config[key].update(value)
            else:
                self._config[key] = value

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Obtém um valor de configuração usando dot notation.
        
        Args:
            key_path: Caminho da chave (ex: 'api.port')
            default: Valor padrão se não encontrado
            
        Returns:
            Valor da configuração ou default
        """
        keys = key_path.split('.')
        current = self._config
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default

    def set(self, key_path: str, value: Any) -> None:
        """
        Define um valor de configuração usando dot notation.
        
        Args:
            key_path: Caminho da chave (ex: 'logging.level')
            value: Valor a ser definido
        """
        keys = key_path.split('.')
        current = self._config
        
        for i, key in enumerate(keys[:-1]):
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
        self.logger.debug(f"Configuração atualizada: {key_path} = {value}")

    def save(self, file_path: str, format: str = 'json') -> None:
        """
        Salva as configurações atuais em um arquivo.
        
        Args:
            file_path: Caminho de destino
            format: Formato ('json' ou 'yaml')
        """
        path = Path(file_path)
        try:
            with open(path, 'w') as f:
                if format == 'json':
                    json.dump(self._config, f, indent=2)
                elif format in ('yaml', 'yml'):
                    yaml.dump(self._config, f)
                else:
                    raise ValueError(f"Formato inválido: {format}")
            
            self.logger.info(f"Configurações salvas em {file_path}")
        except Exception as e:
            self.logger.error(f"Erro ao salvar configurações: {str(e)}")

    def __str__(self) -> str:
        """Representação segura das configurações (omite valores sensíveis)."""
        hidden_keys = {'password', 'secret', 'connection_string'}
        
        def filter_config(config: Any) -> Any:
            if isinstance(config, dict):
                return {k: '*****' if k in hidden_keys else filter_config(v) 
                       for k, v in config.items()}
            return config
            
        return json.dumps(filter_config(self._config), indent=2)

config = ConfigManager()

DEFAULT_CONFIG_YAML = """
# Configurações do Modelo
model:
  path: "models/production/best_model.joblib"
  threshold: 0.7
  retrain_frequency: "weekly"

# Configurações de Features
features:
  sqli:
    min_keyword_length: 3
    max_query_length: 10000
  text:
    max_features: 100
    svd_components: 20

# Configurações da API
api:
  enabled: true
  host: "0.0.0.0"
  port: 8000
  rate_limit: "100/minute"
"""