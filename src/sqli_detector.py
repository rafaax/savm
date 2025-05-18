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
            'has_union_fragments', 'has_oracle_exploits', 'has_char_encoding', 'has_system_tables', 
            'has_time_delay_fn', 'has_load_file_fn', 'has_delete', 'has_truncate', 'has_alter', 'has_update', 
            'has_insert' 
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
            
            df_features_extracted = df_features_extracted.dropna(subset=['label'])

            queries_for_tfidf = df_features_extracted['query'].astype(str) # Mantenha esta linha explícita

            # Verifica se todas as features customizadas esperadas foram geradas
            missing_custom_features = [f for f in self.custom_feature_names if f not in df_features_extracted.columns]

            if missing_custom_features:
                raise ValueError(f"SQLIFeatureExtractor não gerou as seguintes features esperadas: {missing_custom_features}. Features geradas: {df_features_extracted.columns.tolist()}")

            custom_features_data = df_features_extracted[self.custom_feature_names]
            
            # Fit e transform TF-IDF na 'query' original
            X_tfidf_matrix = self.vectorizer.fit_transform(df_features_extracted['query'].astype(str))

            print("TAMANHO DO VOCABULÁRIO TF-IDF:", len(self.vectorizer.vocabulary_))
            # Se o tamanho for > 0, imprima alguns itens para ver o que ele capturou
            if len(self.vectorizer.vocabulary_) > 0:
                print("ALGUNS ITENS DO VOCABULÁRIO TF-IDF (primeiros 10):")
                count = 0
                for term, index in self.vectorizer.vocabulary_.items():
                    print(f"Termo: '{term}', Índice: {index}")
                    count += 1
                    if count >= 10:
                        break
            else:
                print("VOCABULÁRIO TF-IDF ESTÁ VAZIO.")

            


            self.tfidf_feature_names = self.vectorizer.get_feature_names_out().tolist()

            print("Algumas features TF-IDF:", self.tfidf_feature_names[:20]) # Veja as primeiras
            print("Total de features TF-IDF:", len(self.tfidf_feature_names))

            print("TF-IDF FEATURE NAMES (get_feature_names_out):", self.tfidf_feature_names) # Repetir o print

            X_tfidf_matrix = self.vectorizer.fit_transform(df_features_extracted['query'].astype(str))
            self.tfidf_feature_names = self.vectorizer.get_feature_names_out().tolist()
            X_tfidf_df = pd.DataFrame(X_tfidf_matrix.toarray(), columns=self.tfidf_feature_names, index=df_features_extracted.index)

            # Use as features customizadas NÃO escaladas para a concatenação
            custom_features_data_unscaled = df_features_extracted[self.custom_feature_names]

            X_combined_unscaled = pd.concat([X_tfidf_df, custom_features_data_unscaled], axis=1)
            self.all_feature_names_ordered = X_combined_unscaled.columns.tolist()

            scaled_X_combined_array = self.scaler.fit_transform(X_combined_unscaled)
            X_combined = pd.DataFrame(scaled_X_combined_array, columns=self.all_feature_names_ordered, index=X_combined_unscaled.index)

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
        """Prepara as features para uma única string de query para predição."""

        if not self._is_trained:
            raise RuntimeError("O modelo deve ser treinado antes de fazer predições.")

        # 1. Criar um DataFrame para a query única (necessário para SQLIFeatureExtractor)
    
        single_query_df_input = pd.DataFrame({'query': [query_string]})

        # 2. Extrair features customizadas
        
        df_features_extracted_single = self.feature_extractor.extract_features(single_query_df_input.copy()) # Passar cópia

        custom_features_single_data_unscaled = df_features_extracted_single[self.custom_feature_names]

        # 3. Transformar a query original com o TF-IDF Vectorizer treinado
        
        X_tfidf_matrix_single = self.vectorizer.transform([query_string]) # Passar como lista
        
        X_tfidf_df_single = pd.DataFrame(X_tfidf_matrix_single.toarray(), columns=self.tfidf_feature_names, index=custom_features_single_data_unscaled.index) # Manter o mesmo índice

        # 4. Concatenar features TF-IDF e features customizadas (NÃO ESCALADAS)
        
        X_combined_unscaled_single = pd.concat([X_tfidf_df_single, custom_features_single_data_unscaled], axis=1)
        
        X_combined_unscaled_single = X_combined_unscaled_single[self.all_feature_names_ordered]

        # 5. Escalar o conjunto combinado usando o scaler TREINADO

        scaled_X_combined_single_array = self.scaler.transform(X_combined_unscaled_single)

        scaled_X_combined_single_df = pd.DataFrame(scaled_X_combined_single_array, columns=self.all_feature_names_ordered, index=X_combined_unscaled_single.index)
        
        return scaled_X_combined_single_df



    def predict_single(self, query_string: str):
        if not self._is_trained:
            raise RuntimeError("Modelo não treinado. Por favor, treine o modelo primeiro.")

        # Preparar features
        prepared_features_df = self._prepare_single_query_features(query_string)
        
        # Fazer a predição
        # O método predict do SVC espera um array 2D, mesmo para uma única amostra.
        # O DataFrame já está no formato correto (1 linha, N colunas)
        prediction = self.model.predict(prepared_features_df)
        probability = self.model.predict_proba(prepared_features_df)
        
        return {
            "query": query_string,
            "is_malicious": bool(prediction[0]),
            "probability_benign": probability[0][0],
            "probability_malicious": probability[0][1],
            "label": int(prediction[0])
        }
    


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