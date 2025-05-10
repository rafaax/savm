import re
import numpy as np
import json
from typing import Dict, List, Union, Any
import pandas as pd
from pathlib import Path
import logging

class SQLIFeatureExtractor:
    """
    Extrai características especializadas para detecção de SQL Injection.
    
    Features implementadas:
    - Padrões SQLi em queries
    - Características lexicais
    - Padrões estruturais
    - Análise de tokens
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa o extrator com configurações personalizáveis.
        
        Args:
            config: Dicionário de configuração para thresholds e padrões
        """
        self._setup_default_config()
        if config:
            self.config.update(config)
            
        self._compile_patterns()
        self.logger = logging.getLogger(__name__)
    
    def _setup_default_config(self):
        """Configura padrões e thresholds padrão."""
        self.config = {
            'min_keyword_length': 3,
            'max_query_length': 10000,
            'keyword_patterns': {
                'sql_commands': [
                    'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 
                    'UNION', 'EXEC', 'EXECUTE', 'TRUNCATE', 'ALTER',
                    'CREATE', 'SHOW', 'DESCRIBE', 'USE', 'FROM'
                ],
                'operators': [
                    '=', '>', '<', '>=', '<=', '!=', '<>', 
                    'AND', 'OR', 'NOT', 'LIKE', 'IN', 'BETWEEN'
                ],
                'functions': [
                    'CONCAT', 'SUBSTRING', 'CAST', 'COUNT', 'SUM', 
                    'AVG', 'MAX', 'MIN', 'VERSION', 'DATABASE'
                ],
                'comments': ['--', '/*', '*/', '#'],
                'clauses': ['WHERE', 'GROUP BY', 'ORDER BY', 'HAVING', 'LIMIT']
            },
            'structural_patterns': {
                'quote_pairs': [ ("'", "'"), ('"', '"') ],
                'parentheses': [ ('(', ')') ],
                'semicolon': [';'],
                'encoding': ['%20', '%27', '%3B', '0x']
            }
        }
    
    def _compile_patterns(self):
        """Compila expressões regulares para melhor performance."""
        self.patterns = {
            # Padrões básicos
            'single_quotes': re.compile(r"'"),
            'double_quotes': re.compile(r'"'),
            'comments': re.compile(r'--|\/\*|\*\/|#'),
            'semicolon': re.compile(r';'),
            'parentheses': re.compile(r'$$|$$'),
            'equals': re.compile(r'='),
            'hex': re.compile(r'0x[0-9a-fA-F]+'),
            
            # Padrões avançados SQLi
            'union_pattern': re.compile(r'\bUNION\b.*\bSELECT\b', re.IGNORECASE),
            'tautology': re.compile(r'\bOR\b.*\d+=\d+', re.IGNORECASE),
            'piggyback': re.compile(r';.*\b(DROP|ALTER|CREATE|TRUNCATE)\b', re.IGNORECASE),
            'time_delay': re.compile(r'\b(SLEEP|WAITFOR|BENCHMARK)\b', re.IGNORECASE),
            'information_schema': re.compile(r'INFORMATION_SCHEMA', re.IGNORECASE)
        }
        
        # Compila padrões de keywords
        for category, keywords in self.config['keyword_patterns'].items():
            pattern = '|'.join([f'\b{k}\b' for k in keywords])
            self.patterns[f'kw_{category}'] = re.compile(pattern, re.IGNORECASE)
    
    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extrai features de um DataFrame contendo queries.
        
        Args:
            df: DataFrame contendo coluna 'query' com as consultas SQL
            
        Returns:
            DataFrame original com colunas adicionais de features
        """
        if 'query' not in df.columns:
            raise ValueError("DataFrame deve conter coluna 'query'")
            
        df = df.copy()
        queries = df['query'].astype(str)
        
        # Features básicas
        df['query_length'] = queries.str.len()
        df['space_count'] = queries.str.count(r'\s')
        
        # Features de caracteres especiais
        for char_type in ['single_quotes', 'double_quotes', 'comments', 
                         'semicolon', 'equals', 'parentheses']:
            df[f'has_{char_type}'] = queries.apply(
                lambda x: bool(self.patterns[char_type].search(x))
            ).astype(int)
        
        # Contagem de padrões SQL
        for category in self.config['keyword_patterns']:
            df[f'kw_{category}_count'] = queries.str.count(
                self.patterns[f'kw_{category}']
            )
        
        # Padrões de SQL Injection
        sql_patterns = [
            'union_pattern', 'tautology', 'piggyback', 
            'time_delay', 'information_schema', 'hex'
        ]
        for pattern in sql_patterns:
            df[f'sqli_{pattern}'] = queries.apply(
                lambda x: bool(self.patterns[pattern].search(x))
            ).astype(int)
        
        # Features estruturais avançadas
        df['has_encoding'] = queries.str.contains(
            '|'.join(self.config['structural_patterns']['encoding'])
        ).astype(int)
        
        # Balanceamento de parênteses/aspas
        df['unbalanced_parentheses'] = queries.apply(
            lambda x: abs(x.count('(') - x.count(')'))
        )
        
        # Entropia da query (medida de aleatoriedade)
        df['query_entropy'] = queries.apply(self._calculate_entropy)
        
        self.logger.info(f"Extraídas {len(df.columns) - 1} features")
        return df
    
    def _calculate_entropy(self, text: str) -> float:
        """Calcula entropia de Shannon para a string."""
        if not text:
            return 0.0
            
        text = text.lower()
        prob = [text.count(c) / len(text) for c in set(text)]
        return -sum(p * np.log2(p) for p in prob if p > 0)
    
    def save_features_config(self, output_path: Union[str, Path]):
        """Salva a configuração de features para reprodução futura."""
        config = {
            'feature_names': self.get_feature_names(),
            'config': self.config
        }
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def load_features_config(cls, config_path: Union[str, Path]):
        """Carrega configuração a partir de arquivo."""
        with open(config_path) as f:
            config = json.load(f)
        return cls(config['config'])
    
    def get_feature_names(self) -> List[str]:
        """Retorna lista de nomes de todas as features extraídas."""
        return [
            'query_length', 'space_count',
            *[f'has_{c}' for c in [
                'single_quotes', 'double_quotes', 'comments', 
                'semicolon', 'equals', 'parentheses'
            ]],
            *[f'kw_{c}_count' for c in self.config['keyword_patterns']],
            *[f'sqli_{p}' for p in [
                'union_pattern', 'tautology', 'piggyback', 
                'time_delay', 'information_schema', 'hex'
            ]],
            'has_encoding',
            'unbalanced_parentheses',
            'query_entropy'
        ]