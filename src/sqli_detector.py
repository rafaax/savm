import pandas as pd
from src.features_extractor import SQLIFeatureExtractor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

class SQLIDetector:
    def __init__(self):
        self.feature_extractor = SQLIFeatureExtractor()  # Inicializa o extrator de features
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.scaler = MinMaxScaler()
        self.model = SVC(kernel='rbf', C=1.5, gamma='scale', class_weight='balanced')
        self.test_indices = None  # Para armazenar os índices de teste

    def train(self, df):
        """Treina o modelo completo"""
        try:
            # Verifica se o DataFrame tem a estrutura esperada
            if 'query' not in df.columns or 'label' not in df.columns:
                raise ValueError("DataFrame deve conter colunas 'query' e 'label'")
                
            # Extração de features
            df = self.feature_extractor.extract_features(df)
            
            # Seleciona features para o modelo
            feature_columns = [
                'has_or', 'has_and', 'has_not', 'has_union', 'has_select_all',
                'has_comment', 'has_equals', 'has_drop', 'has_exec',
                'has_function_exploit', 'has_single_quote', 'has_parentheses',
                'has_semicolon', 'has_concat_symbols', 'has_hex', 'has_encoding',
                'has_system_tables', 'query_length', 'space_count', 'quote_count',
                'special_char_count', 'has_union_fragments', 'has_oracle_exploits',
                'has_char_encoding', 'has_time_delay_fn', 'has_load_file_fn'
            ]
            
            # Verifica se todas as features existem
            missing_features = [f for f in feature_columns if f not in df.columns]
            if missing_features:
                raise ValueError(f"Features ausentes: {missing_features}")
            
            custom_features = df[feature_columns]
            X_tfidf = self.vectorizer.fit_transform(df['query'])
            
            # Combina features
            X = pd.concat([
                pd.DataFrame(X_tfidf.toarray(), 
                           columns=self.vectorizer.get_feature_names_out()),
                pd.DataFrame(self.scaler.fit_transform(custom_features), 
                           columns=custom_features.columns)
            ], axis=1)
            
            # Treinamento com preservação de índices
            X_train, X_test, y_train, y_test = train_test_split(
                X, df['label'], test_size=0.3, random_state=42
            )
            
            # Armazena os índices originais para análise posterior
            self.test_indices = X_test.index
            
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
            
            return y_test, y_pred
            
        except Exception as e:
            print(f"Erro durante o treinamento: {str(e)}")
            raise


