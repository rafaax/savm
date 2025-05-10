import re
import numpy as np
import pandas as pd
from typing import Dict, List, Union
from pathlib import Path
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

class TextFeatureExtractor:
    """
    Extrai características gerais de texto para análise de consultas SQL.
    
    Features incluídas:
    - Estatísticas básicas de texto
    - Caracteres especiais e padrões
    - Complexidade lexical
    - Features semânticas (via redução dimensional)
    """
    
    def __init__(self, max_features=100):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.feature_names = []
        
    def _setup_default_config(self):
        """Define configurações padrão."""
        self.config = {
            'ngram_range': (1, 2),
            'max_features': 100,
            'svd_components': 10,
            'special_chars': [';', '\'', '"', '--', '/*', '*/', '#', '@', '=', '<', '>'],
            'keyword_patterns': {
                'sql': [
                    'SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE', 
                    'DELETE', 'JOIN', 'GROUP BY', 'ORDER BY'
                ],
                'operators': ['AND', 'OR', 'NOT', 'IN', 'LIKE', 'BETWEEN']
            }
        }
        
    def _initialize_models(self):
        """Inicializa modelos para features avançadas."""
        self.tfidf = TfidfVectorizer(
            ngram_range=self.config['ngram_range'],
            max_features=self.config['max_features'],
            stop_words=None,
            analyzer='char'  # Analisa caracteres para padrões SQL
        )
        
        self.svd = TruncatedSVD(
            n_components=self.config['svd_components'],
            random_state=42
        )
        
    def extract(self, df: pd.DataFrame, fit_models: bool = False) -> pd.DataFrame:
        if fit_models:
            tfidf_matrix = self.vectorizer.fit_transform(df['query'])
            self.feature_names = self.vectorizer.get_feature_names_out()
        else:
            tfidf_matrix = self.vectorizer.transform(df['query'])
        
        # Garante que temos o mesmo número de features
        features = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=self.feature_names
        )
        return pd.concat([df, features], axis=1)
    
    def _extract_basic_features(self, df: pd.DataFrame, queries: pd.Series) -> pd.DataFrame:
        """Extrai features básicas de texto."""
        # Estatísticas de comprimento
        df['text_length'] = queries.str.len()
        df['word_count'] = queries.str.split().str.len()
        df['avg_word_length'] = queries.apply(
            lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0
        )
        
        # Contagem de caracteres especiais
        for char in self.config['special_chars']:
            df[f'char_{char}_count'] = queries.str.count(re.escape(char))
        
        # Contagem de palavras-chave SQL
        for category, keywords in self.config['keyword_patterns'].items():
            pattern = '|'.join([f'\b{k}\b' for k in keywords])
            df[f'kw_{category}_count'] = queries.str.count(pattern, flags=re.IGNORECASE)
        
        # Diversidade lexical
        df['lexical_diversity'] = queries.apply(
            lambda x: len(set(x.split())) / len(x.split()) if x.split() else 0
        )
        
        return df
    
    def _extract_advanced_features(self, df: pd.DataFrame, queries: pd.Series, fit: bool) -> pd.DataFrame:
        """Extrai features avançadas usando TF-IDF e SVD."""
        try:
            # Features de padrões de caracteres via TF-IDF
            if fit:
                tfidf_features = self.tfidf.fit_transform(queries)
                svd_features = self.svd.fit_transform(tfidf_features)
            else:
                tfidf_features = self.tfidf.transform(queries)
                svd_features = self.svd.transform(tfidf_features)
            
            # Adiciona componentes SVD como features
            for i in range(self.svd.n_components):
                df[f'svd_component_{i}'] = svd_features[:, i]
                
            return df
            
        except Exception as e:
            self.logger.error(f"Erro ao extrair features avançadas: {str(e)}")
            return df
    
    def get_feature_names(self) -> List[str]:
        """Retorna lista de nomes de todas as features."""
        basic_features = [
            'text_length', 'word_count', 'avg_word_length', 'lexical_diversity',
            *[f'char_{c}_count' for c in self.config['special_chars']],
            *[f'kw_{cat}_count' for cat in self.config['keyword_patterns']]
        ]
        
        advanced_features = [
            f'svd_component_{i}' for i in range(self.config['svd_components'])
        ]
        
        return basic_features + advanced_features
    
    def save_models(self, output_dir: Union[str, Path]):
        """Salva os modelos TF-IDF e SVD para uso futuro."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        import joblib
        joblib.dump(self.tfidf, output_dir / 'tfidf_model.joblib')
        joblib.dump(self.svd, output_dir / 'svd_model.joblib')
        
    def load_models(self, model_dir: Union[str, Path]):
        """Carrega modelos pré-treinados."""
        import joblib
        self.tfidf = joblib.load(Path(model_dir) / 'tfidf_model.joblib')
        self.svd = joblib.load(Path(model_dir) / 'svd_model.joblib')