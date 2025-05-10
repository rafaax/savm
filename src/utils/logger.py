import logging
from logging.handlers import RotatingFileHandler, SMTPHandler
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime
from utils.config import config

class JSONFormatter(logging.Formatter):
    """Formatador de logs em JSON para integração com sistemas ELK."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Formata o log como JSON estruturado."""
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'message': record.getMessage(),
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.threadName,
        }
        
        # Adiciona contexto adicional se existir
        if hasattr(record, 'context'):
            log_data.update(record.context)
            
        return json.dumps(log_data, ensure_ascii=False)

class ContextFilter(logging.Filter):
    """Filtro que adiciona contexto comum a todos os logs."""
    
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.context = context or {}
        
    def filter(self, record: logging.LogRecord) -> bool:
        """Adiciona contexto ao registro de log."""
        for key, value in self.context.items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True

def setup_logger(name: str = 'sqli_detector') -> logging.Logger:
    """
    Configura um logger com handlers para arquivo, console e email.
    
    Args:
        name: Nome do logger
        
    Returns:
        Instância do logger configurado
    """
    # Cria o logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Remove handlers existentes para evitar duplicação
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Adiciona filtro de contexto
    context = {
        'app_version': config.get('app.version', '1.0.0'),
        'environment': config.get('environment', 'development')
    }
    logger.addFilter(ContextFilter(context))
    
    # Configuração básica
    log_file = Path(config.get('logging.file', 'logs/sqli_detector.log'))
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Handler para arquivo (rotativo)
    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=config.get('logging.max_size', 10 * 1024 * 1024),  # 10MB
        backupCount=config.get('logging.backup_count', 5),
        encoding='utf-8'
    )
    file_handler.setFormatter(JSONFormatter())
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    # Handler para console (apenas em desenvolvimento)
    if config.get('environment') == 'development':
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        console_handler.setLevel(logging.DEBUG)
        logger.addHandler(console_handler)
    
    # Handler para email (apenas em produção para erros críticos)
    if (config.get('environment') == 'production' and 
        config.get('logging.email.enabled', False)):
        mail_handler = SMTPHandler(
            mailhost=(config.get('logging.email.host'), 
                     config.get('logging.email.port')),
            fromaddr=config.get('logging.email.from'),
            toaddrs=config.get('logging.email.to'),
            subject='SQLi Detector - Erro Crítico',
            credentials=(config.get('logging.email.username'),
                        config.get('logging.email.password')),
            secure=()
        )
        mail_handler.setLevel(logging.ERROR)
        mail_handler.setFormatter(logging.Formatter('''
            Message type:       %(levelname)s
            Location:           %(pathname)s:%(lineno)d
            Module:             %(module)s
            Function:           %(funcName)s
            Time:               %(asctime)s

            Message:

            %(message)s
        '''))
        logger.addHandler(mail_handler)
    
    return logger

def log_operation(
    logger: logging.Logger,
    operation: str,
    level: str = 'info',
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs
) -> None:
    """
    Registra uma operação com metadados estruturados.
    
    Args:
        logger: Instância do logger
        operation: Nome da operação
        level: Nível do log ('debug', 'info', 'warning', 'error', 'critical')
        metadata: Dicionário com metadados adicionais
        kwargs: Atributos adicionais para o log
    """
    log_method = getattr(logger, level.lower(), logger.info)
    
    extra = {
        'operation': operation,
        'context': metadata or {},
        **kwargs
    }
    
    # Cria um LogRecord com contexto adicional
    log_method(operation, extra=extra)

# Logger global pré-configurado
logger = setup_logger()

# Exemplo de uso avançado
if __name__ == '__main__':
    # Exemplo 1: Log simples
    logger.info("Sistema iniciado")
    
    # Exemplo 2: Log com contexto
    log_operation(
        logger,
        "detection",
        level="info",
        metadata={
            "query": "SELECT * FROM users",
            "is_sqli": False,
            "probability": 0.15
        },
        execution_time=0.042
    )
    
    # Exemplo 3: Log de erro com stack trace
    try:
        1 / 0
    except Exception as e:
        logger.error(
            "Erro na divisão",
            exc_info=True,
            extra={
                'context': {
                    'var1': 10,
                    'var2': 0
                }
            }
        )