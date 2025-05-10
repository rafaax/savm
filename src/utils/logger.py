import logging
from logging.handlers import RotatingFileHandler
import sys
from pathlib import Path
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """Formatador de logs em JSON para melhor integração"""
    
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'message': record.getMessage(),
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if hasattr(record, 'context'):
            log_data['context'] = record.context
            
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_data, ensure_ascii=False)

def setup_logger():
    """Configuração principal do logger"""
    
    # Cria diretório de logs se não existir
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger('sqli_detector')
    logger.setLevel(logging.DEBUG)
    
    # Remove handlers existentes
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Configuração do handler de arquivo
    file_handler = RotatingFileHandler(
        filename=log_dir / 'detector.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(JSONFormatter())
    file_handler.setLevel(logging.INFO)
    
    # Configuração do handler de console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s'
    ))
    console_handler.setLevel(logging.DEBUG)
    
    # Adiciona handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def log_with_context(message: str, level: str = 'info', **context):
    """
    Registra mensagem com contexto adicional
    
    Args:
        message: Mensagem principal
        level: Nível do log (debug, info, warning, error, critical)
        context: Dicionário com metadados adicionais
    """
    logger = logging.getLogger('sqli_detector')
    log_func = getattr(logger, level.lower(), logger.info)
    
    extra = {'context': context}
    log_func(message, extra=extra)

# Logger global
logger = setup_logger()

# Exemplos de uso:
if __name__ == '__main__':
    # Log básico
    logger.info("Sistema de detecção inicializado")
    
    # Log com contexto
    log_with_context(
        "Tentativa de SQLi detectada",
        level="warning",
        query="SELECT * FROM users WHERE 1=1",
        probability=0.92,
        source_ip="192.168.1.100"
    )
    
    # Log de erro com stacktrace
    try:
        raise ValueError("Exemplo de erro")
    except Exception as e:
        logger.error("Erro na análise", exc_info=True)