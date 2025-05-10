import pandas as pd
import joblib
import os
import traceback
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
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
        self._last_evaluation_data = None
        


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
            X_tfidf_matrix = self.vectorizer.fit_transform(df_features_extracted['query'].astype(str))
            self.tfidf_feature_names = self.vectorizer.get_feature_names_out().tolist()

            X_tfidf_df = pd.DataFrame(X_tfidf_matrix.toarray(), columns=self.tfidf_feature_names, index=df_features_extracted.index)

            # Fit e transform Scaler nas features customizadas
            scaled_custom_features_array = self.scaler.fit_transform(custom_features_data)
            scaled_custom_features_df = pd.DataFrame(scaled_custom_features_array, columns=self.custom_feature_names, index=custom_features_data.index)
            
            X_combined = pd.concat([X_tfidf_df, scaled_custom_features_df], axis=1)
            self.all_feature_names_ordered = X_combined.columns.tolist()

            y = df_features_extracted['label']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_combined, y, test_size=0.3, random_state=42, stratify=y
            )
            
            self.test_indices = X_test.index.tolist()
            
            self.model.fit(X_train, y_train)
            y_pred_test = self.model.predict(X_test)
            
            self._is_trained = True

            # Preparar dados para get_last_evaluation_data
            df_original_test_set = self.df_cache.loc[self.test_indices].copy() # Usa self.df_cache e self.test_indices
            self._last_evaluation_data = (df_original_test_set, y_test.values, y_pred_test, self.test_indices) # y_test.values para array numpy

            # Calcular métricas
            accuracy = accuracy_score(y_test, y_pred_test)
            precision = precision_score(y_test, y_pred_test, zero_division=0)
            recall = recall_score(y_test, y_pred_test, zero_division=0)
            f1 = f1_score(y_test, y_pred_test, zero_division=0)
            report_dict = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)
            conf_matrix_list = confusion_matrix(y_test, y_pred_test).tolist()
            
            # retorna todas métricas necessarias para a aplicação
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "classification_report_dict": report_dict,
                "confusion_matrix_list": conf_matrix_list
            }
            
        except Exception as e:

            self._is_trained = False
            print(f"Erro crítico durante o treinamento: {str(e)}")
            traceback.print_exc()
            return {
                "accuracy": None, "precision": None, "recall": None, "f1_score": None,
                "classification_report_dict": None, "confusion_matrix_list": None, "error": str(e)
            }




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

        if not self._is_trained or self._last_evaluation_data is None:
            print("WARN: Modelo não treinado ou dados de avaliação não disponíveis.")
            return None
        return self._last_evaluation_data
    


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