import pandas as pd
import joblib
import os
import traceback
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from src.features_extractor import SQLIFeatureExtractor


class SQLIDetector:
    def __init__(self):
        self.feature_extractor = SQLIFeatureExtractor()
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), lowercase=True)
        self.scaler = MinMaxScaler()
        self.model = SVC(kernel='rbf', C=1.5, gamma='scale', class_weight='balanced', probability=True)

        self._is_trained = False
        self.tfidf_feature_names = []

        self.custom_feature_names = [
            'query_length', 'has_or', 'has_and', 'has_not', 'has_union', 'has_select_all',
            'has_comment', 'has_equals', 'has_drop', 'has_exec', 'has_function_exploit',
            'has_single_quote', 'has_parentheses', 'has_semicolon', 'has_concat_symbols',
            'has_hex', 'has_encoding', 'space_count', 'quote_count', 'special_char_count',
            'has_union_fragments', 'has_oracle_exploits', 'has_char_encoding',
            'has_system_tables', 'has_time_delay_fn', 'has_load_file_fn'
        ]
        self.all_feature_names_ordered = []

        self.test_indices = None
        self.y_test_cache = None
        self.y_pred_cache = None
        self.df_cache = None


    def train(self, df_input: pd.DataFrame):
        """Treina o modelo completo e armazena informações necessárias para predição e análise."""
        
        try:
            if 'query' not in df_input.columns or 'label' not in df_input.columns:
                raise ValueError("DataFrame de entrada deve conter colunas 'query' e 'label'")

            # Faz uma cópia para não modificar o DataFrame original passado para a função
            df_processed = df_input.copy()
            self.df_cache = df_input.copy() # Armazena uma cópia do DF original não processado

            # Extrai features
            df_features_extracted = self.feature_extractor.extract_features(df_processed)

            # Verifica se todas as features customizadas esperadas foram geradas
            missing_custom_features = [f for f in self.custom_feature_names if f not in df_features_extracted.columns]

            if missing_custom_features:
                raise ValueError(f"SQLIFeatureExtractor não gerou as seguintes features esperadas: {missing_custom_features}. Features geradas: {df_features_extracted.columns.tolist()}")

            custom_features_data = df_features_extracted[self.custom_feature_names]
            
            # Fit e transform TF-IDF na 'query' original
            X_tfidf_matrix = self.vectorizer.fit_transform(df_features_extracted['query'])
            self.tfidf_feature_names = self.vectorizer.get_feature_names_out().tolist()

            # Preserva o índice original do df_features_extracted
            X_tfidf_df = pd.DataFrame(X_tfidf_matrix.toarray(), columns=self.tfidf_feature_names, index=df_features_extracted.index)

            # Fit e transform Scaler nas features customizadas
            scaled_custom_features_array = self.scaler.fit_transform(custom_features_data)

            # Preserva o índice e nomes das colunas customizadas
            scaled_custom_features_df = pd.DataFrame(scaled_custom_features_array, columns=self.custom_feature_names, index=custom_features_data.index)
            
            # Combina features TF-IDF e customizadas escalonadas
            X_combined = pd.concat([X_tfidf_df, scaled_custom_features_df], axis=1)
            self.all_feature_names_ordered = X_combined.columns.tolist() # Armazena a ordem final das features

            y = df_features_extracted['label']
            
            # Treinamento com preservação de índices para X_test, y_test
            X_train, X_test, y_train, y_test = train_test_split(
                X_combined, y, test_size=0.3, random_state=42, stratify=y
            )
            
            self.test_indices = X_test.index # df_features_extracted
            
            self.model.fit(X_train, y_train)
            y_pred_test = self.model.predict(X_test)
            
            self._is_trained = True
            self.y_test_cache = y_test
            self.y_pred_cache = y_pred_test
            
            accuracy = accuracy_score(y_test, y_pred_test)
            
            report_dict = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0) # zero_division=0 para evitar warnings se uma classe não tem predições/verdades no teste
            conf_matrix = confusion_matrix(y_test, y_pred_test).tolist()

            print(f"\nModelo Treinado. Acurácia: {accuracy:.4f}")
            print("\nClassification Report (Treinamento):")
            print(classification_report(y_test, y_pred_test, zero_division=0))
            print("\nConfusion Matrix (Treinamento):")
            print(confusion_matrix(y_test, y_pred_test))
            
            return {
                "accuracy": accuracy,
                "classification_report": report_dict,
                "confusion_matrix": conf_matrix
            }
            
        except Exception as e:
            self._is_trained = False
            print(f"Erro crítico durante o treinamento: {str(e)}")
            traceback.print_exc()
            raise



    def _prepare_single_query_features(self, query_string: str) -> pd.DataFrame:
        """Prepara as features para uma única string de query."""

        if not self._is_trained:
            raise RuntimeError("Modelo não treinado")

        # 1. Criar DataFrame para a query
        query_df_single_row = pd.DataFrame({'query': [query_string]})

        # 2. Extrair features customizadas usando seu extrator
        custom_features_extracted_df = self.feature_extractor.extract_features(query_df_single_row.copy()) # Passa uma cópia
        
        missing_custom_features = [f for f in self.custom_feature_names if f not in custom_features_extracted_df.columns] # Verifica se todas as features customizadas esperadas foram geradas

        if missing_custom_features:
            raise ValueError(f"SQLIFeatureExtractor não gerou as seguintes features esperadas para a query: {missing_custom_features}")
        
        
        custom_features_single_data = custom_features_extracted_df[self.custom_feature_names] # Seleciona as features customizadas na ordem definida

        # 3. Aplicar TF-IDF (transform) na query
        tfidf_matrix_single = self.vectorizer.transform(custom_features_extracted_df['query'])
        tfidf_df_single = pd.DataFrame(tfidf_matrix_single.toarray(), columns=self.tfidf_feature_names)

        # 4. Aplicar Scaler (transform) nas features customizadas
        scaled_custom_features_array_single = self.scaler.transform(custom_features_single_data)
        scaled_custom_features_df_single = pd.DataFrame(scaled_custom_features_array_single, columns=self.custom_feature_names)

        # 5. Combinar features TF-IDF e customizadas escalonadas
        combined_features_single = pd.concat([tfidf_df_single, scaled_custom_features_df_single], axis=1)
        
        # 6. Garantir a mesma ordem de colunas que no treinamento
        for col_name in self.all_feature_names_ordered:
            if col_name not in combined_features_single:
                combined_features_single[col_name] = 0.0 # Importante usar float para consistência
        
        # Reordenar para corresponder à ordem do treinamento
        return combined_features_single[self.all_feature_names_ordered]



    def predict_single(self, query_string: str) -> int:
        """Prevê o label para uma única string de query."""
        prepared_features = self._prepare_single_query_features(query_string)
        prediction = self.model.predict(prepared_features)
        return int(prediction[0])
    


    def predict_proba_single(self, query_string: str) -> np.ndarray:
        """Prevê as probabilidades para uma única string de query."""
        if not hasattr(self.model, "predict_proba") or not self.model.probability:
             raise TypeError("O modelo SVC não foi configurado com probability=True.")
        prepared_features = self._prepare_single_query_features(query_string)
        probabilities = self.model.predict_proba(prepared_features)
        return probabilities[0]
    


    def is_trained(self) -> bool:
        return self._is_trained



    def get_last_evaluation_data(self):
        """Retorna os dados da última avaliação para análise de falsos negativos."""

        if not self._is_trained or self.y_test_cache is None or self.y_pred_cache is None or self.test_indices is None or self.df_cache is None:
            return None
        return self.df_cache, self.y_test_cache, self.y_pred_cache, self.test_indices
    


    def save_model(self, filepath: str):
        """Salva a instância inteira do detector."""

        if not self._is_trained:
            print("Aviso: Tentando salvar um modelo que ainda não foi treinado.")
        
        try:
            dir_name = os.path.dirname(filepath)
            if dir_name: # Se filepath incluir um diretório
                os.makedirs(dir_name, exist_ok=True)
            
            joblib.dump(self, filepath)
            print(f"Detector SQLi salvo com sucesso em: {filepath}")
        except Exception as e:
            print(f"Erro ao salvar o detector SQLi: {str(e)}")
            raise



    @classmethod
    def load_model(cls, filepath: str):
        """Carrega uma instância salva do detector SQLi."""

        try:
            if not os.path.exists(filepath):
                print(f"Arquivo do modelo não encontrado em: {filepath}. Retornando None.")
                return None
            
            detector = joblib.load(filepath)

            # garantir que é uma instância da classe e está treinado
            if isinstance(detector, cls) and detector.is_trained():
                print(f"Detector SQLi carregado com sucesso de: {filepath}")
                return detector
            
            else:
                print(f"Arquivo carregado de {filepath} não é um detector SQLi treinado válido ou é de um tipo inesperado.")
                return None
            
        except Exception as e:
            print(f"Erro ao carregar o detector SQLi de {filepath}: {str(e)}. Retornando None.")
            return None