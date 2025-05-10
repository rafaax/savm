import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from utils.config import config
import logging

class SQLiteManager:
    """
    Gerenciador de conexão com banco SQLite para o sistema de detecção de SQLi.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Args:
            db_path: Caminho para o arquivo SQLite (None usa config padrão)
        """
        self.logger = logging.getLogger(__name__)
        self.db_path = Path(db_path) if db_path else Path(config.get('database.path', 'data/sqli_detector.db'))
        self.conn = None
        self._initialize()

    def _initialize(self):
        """Cria a conexão e garante a estrutura do banco."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Cria conexão
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        
        # Cria tabelas se não existirem
        self._create_tables()
        self.logger.info(f"Banco SQLite conectado: {self.db_path}")

    def _create_tables(self):
        """Cria a estrutura inicial do banco."""
        queries = [
            """CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                is_sqli BOOLEAN NOT NULL,
                probability REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                source_ip TEXT,
                user_agent TEXT
            )""",
            
            """CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                path TEXT NOT NULL,
                performance REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT FALSE
            )""",
            
            """CREATE INDEX IF NOT EXISTS idx_queries_timestamp ON queries(timestamp)""",
            """CREATE INDEX IF NOT EXISTS idx_queries_is_sqli ON queries(is_sqli)"""
        ]
        
        try:
            cursor = self.conn.cursor()
            for query in queries:
                cursor.execute(query)
            self.conn.commit()
        except Exception as e:
            self.logger.error(f"Erro ao criar tabelas: {str(e)}")
            raise

    def log_query(self, query_data: Dict[str, Any]) -> int:
        """
        Registra uma consulta analisada no banco.
        
        Args:
            query_data: Dicionário com:
                - query: texto da consulta
                - is_sqli: resultado da detecção
                - probability: probabilidade
                - source_ip: IP de origem (opcional)
                - user_agent: User Agent (opcional)
                
        Returns:
            ID do registro inserido
        """
        query = """
        INSERT INTO queries (query, is_sqli, probability, source_ip, user_agent)
        VALUES (?, ?, ?, ?, ?)
        """
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, (
                query_data['query'],
                int(query_data['is_sqli']),
                query_data['probability'],
                query_data.get('source_ip'),
                query_data.get('user_agent')
            ))
            self.conn.commit()
            return cursor.lastrowid
        except Exception as e:
            self.logger.error(f"Erro ao registrar query: {str(e)}")
            raise

    def get_queries(self, limit: int = 100) -> List[Dict]:
        """Recupera consultas recentes."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM queries ORDER BY timestamp DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"Erro ao buscar queries: {str(e)}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de detecção."""
        try:
            cursor = self.conn.cursor()
            
            # Total de consultas
            cursor.execute("SELECT COUNT(*) FROM queries")
            total = cursor.fetchone()[0]
            
            # Consultas maliciosas
            cursor.execute("SELECT COUNT(*) FROM queries WHERE is_sqli = 1")
            malicious = cursor.fetchone()[0]
            
            # Taxa de detecção
            rate = malicious / total if total > 0 else 0
            
            # Última detecção
            cursor.execute("""
                SELECT query, timestamp 
                FROM queries 
                WHERE is_sqli = 1 
                ORDER BY timestamp DESC 
                LIMIT 1
            """)
            last_detection = cursor.fetchone()
            
            return {
                'total_queries': total,
                'malicious_queries': malicious,
                'detection_rate': round(rate, 4),
                'last_detection': dict(last_detection) if last_detection else None
            }
        except Exception as e:
            self.logger.error(f"Erro ao calcular estatísticas: {str(e)}")
            return {}

    def register_model(self, model_data: Dict[str, Any]) -> int:
        """
        Registra um novo modelo no banco.
        
        Args:
            model_data: Dicionário com:
                - name: nome do modelo
                - version: versão
                - path: caminho do arquivo
                - performance: score de avaliação
                
        Returns:
            ID do modelo registrado
        """
        try:
            # Desativa modelos anteriores
            cursor = self.conn.cursor()
            cursor.execute("UPDATE models SET is_active = FALSE")
            
            # Insere novo modelo
            cursor.execute("""
                INSERT INTO models (name, version, path, performance, is_active)
                VALUES (?, ?, ?, ?, TRUE)
            """, (
                model_data['name'],
                model_data['version'],
                model_data['path'],
                model_data.get('performance')
            ))
            
            self.conn.commit()
            return cursor.lastrowid
        except Exception as e:
            self.logger.error(f"Erro ao registrar modelo: {str(e)}")
            raise

    def get_active_model(self) -> Optional[Dict]:
        """Recupera o modelo ativo."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM models WHERE is_active = TRUE LIMIT 1")
            row = cursor.fetchone()
            return dict(row) if row else None
        except Exception as e:
            self.logger.error(f"Erro ao buscar modelo ativo: {str(e)}")
            return None

    def export_to_dataframe(self, table_name: str) -> Optional[pd.DataFrame]:
        """Exporta uma tabela para DataFrame."""
        try:
            return pd.read_sql(f"SELECT * FROM {table_name}", self.conn)
        except Exception as e:
            self.logger.error(f"Erro ao exportar {table_name}: {str(e)}")
            return None

    def close(self):
        """Fecha a conexão com o banco."""
        if self.conn:
            self.conn.close()
            self.logger.info("Conexão com SQLite encerrada")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# Instância singleton para uso global
database = SQLiteManager()