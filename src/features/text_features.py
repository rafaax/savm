import re
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional
from pathlib import Path
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import joblib

class TextFeatureExtractor:
    """
    Extrator de features para detecção de SQLi com:
    - Features estáticas robustas
    - Vetorização consistente
    - Fallback seguro
    """

    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {
            'max_features': 500,
            'svd_components': 10,
            'special_chars': [';', "'", '"', '--', '/*', '*/', '#', '=', '<', '>', '(', ')', 'OR', 'AND'],
            'keywords': ['SELECT', 'FROM', 'WHERE', 'UNION', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'EXEC']
        }

        # Model persistence paths
        self.models_dir = Path('models/text_features')
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Initialize feature extractors
        self._initialize_pipelines()
        self._load_or_train_models()

    def _initialize_pipelines(self):
        """Configura pipelines com fallback"""
        # Pipeline para features básicas
        self.basic_feature_pipe = Pipeline([
            ('basic_featurizer', self._get_basic_feature_extractor())
        ])

        # Pipeline para features avançadas
        self.advanced_feature_pipe = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=self.config['max_features'],
                token_pattern=r'(?u)\b\w+\b|[-!$%^&*()_+|~=`{}$$$$:";\'<>?,.\/]',
                ngram_range=(1, 2)
            )),
            ('svd', TruncatedSVD(n_components=self.config['svd_components']))
        ])

    def _get_basic_feature_extractor(self):
        """Factory para extrator de features básicas"""
        def extractor(queries: pd.Series) -> pd.DataFrame:
            df = pd.DataFrame(index=queries.index)
            
            # Estatísticas básicas
            df['query_length'] = queries.str.len()
            df['token_count'] = queries.str.split().str.len()
            df['avg_token_length'] = queries.apply(
                lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0
            )

            # Caracteres especiais
            for char in self.config['special_chars']:
                safe_char = char.replace('*', r'\*').replace('+', r'\+')
                df[f'char_{char}_count'] = queries.str.count(re.escape(safe_char))

            # Palavras-chave SQL
            for keyword in self.config['keywords']:
                df[f'kw_{keyword.lower()}'] = queries.str.contains(
                    f'\b{keyword}\b', 
                    case=False, 
                    regex=True
                ).astype(int)

            return df
        return extractor

    def _load_or_train_models(self):
        """Versão robusta com auto-treinamento"""
        try:
            # Tenta carregar modelos existentes
            pipe_path = self.models_dir / 'advanced_pipe.joblib'
            
            if pipe_path.exists():
                self.advanced_feature_pipe = joblib.load(pipe_path)
                self.logger.info("Modelos carregados do cache")
            else:
                raise FileNotFoundError("Modelos não encontrados - treinando...")
                
        except Exception as e:
            self.logger.warning(f"{str(e)}")
            
            # Dados de treinamento robustos
            train_data = [
                # 50 exemplos normais
                "SELECT id FROM users",
                "UPDATE products SET price = 10 WHERE id = 1",
                # ... adicione mais exemplos ...
                
                # 50 exemplos maliciosos
                "' OR 1=1 --",
                "1; SHUTDOWN",
                # ... adicione mais exemplos ...
            ]
            
            try:
                self.advanced_feature_pipe.fit(train_data)
                joblib.dump(self.advanced_feature_pipe, pipe_path)
                self.logger.info("✅ Modelos treinados e salvos com sucesso")
            except Exception as train_error:
                self.logger.critical(f"Falha crítica no treinamento: {str(train_error)}")
                raise


    def _train_with_default_data(self):
        """Treinamento com exemplos balanceados"""
        training_queries = [
            # Queries seguras
            "SELECT * FROM users WHERE id = 1",
            "INSERT INTO products (name) VALUES ('book')",
            "UPDATE accounts SET balance = 100 WHERE user_id = 42",
            "DELETE FROM sessions WHERE expired = true",
            
            # Queries maliciosas
            "' OR 1=1 --",
            "admin'--",
            "1; DROP TABLE users--",
            "' UNION SELECT username, password FROM users--",
            "SELECT * FROM information_schema.tables"
        ]

        try:
            self.advanced_feature_pipe.fit(training_queries)
            joblib.dump(self.advanced_feature_pipe, self.models_dir / 'advanced_pipe.joblib')
        except Exception as e:
            self.logger.error(f"Falha no treinamento: {str(e)}")
            raise

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extrai features de forma robusta com fallback
        """
        try:
            # Features básicas
            basic_features = self.basic_feature_pipe.named_steps['basic_featurizer'](df['query'])
            
            # Features avançadas
            try:
                advanced_features = self.advanced_feature_pipe.transform(df['query'])
                svd_cols = [f'svd_{i}' for i in range(self.config['svd_components'])]
                advanced_df = pd.DataFrame(advanced_features, columns=svd_cols, index=df.index)
            except Exception as e:
                self.logger.warning(f"Falha nas features avançadas: {str(e)}")
                advanced_df = pd.DataFrame(index=df.index)
                
            # Combina todas as features
            return pd.concat([df, basic_features, advanced_df], axis=1).fillna(0)
            
        except Exception as e:
            self.logger.error(f"Falha crítica na extração: {str(e)}")
            return df  # Retorna o DataFrame original como fallback

    def get_feature_names(self) -> List[str]:
        """Retorna todos os nomes de features esperados"""
        basic_features = [
            'query_length', 'token_count', 'avg_token_length',
            *[f'char_{c}_count' for c in self.config['special_chars']],
            *[f'kw_{k.lower()}' for k in self.config['keywords']]
        ]
        
        advanced_features = [
            f'svd_{i}' for i in range(self.config['svd_components'])
        ]
        
        return basic_features + advanced_features