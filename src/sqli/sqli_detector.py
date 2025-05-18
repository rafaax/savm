import pandas as pd
import joblib
import os
import traceback
import numpy as np
import time
from sklearn.inspection import permutation_importance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from src.sqli.features_extractor import SQLIFeatureExtractor


class SQLIDetector:
    def __init__(self):
        self.feature_extractor = SQLIFeatureExtractor()
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),
            lowercase=True,
            stop_words=None,
            analyzer='char_wb'  # capturar padrões como '--' ou '/*' :///
        )
        self.scaler = MinMaxScaler()

        self.model_params = {
            'kernel': 'linear',
            'C': 1.5,
            'gamma': 'scale',
            'class_weight': 'balanced',
            'probability': True
        }

        self.model = SVC(**self.model_params)

        self._is_trained = False
        self.tfidf_feature_names = []

        self.custom_feature_names = [
            # Features básicas
            'query_length', 'space_count', 'special_char_count',
            # Operadores lógicos
            'has_or', 'has_and', 'has_not',
            # Comandos DML/DDL
            'has_drop', 'has_delete', 'has_truncate', 'has_alter',
            'has_unconditional_delete', 'has_destructive_command',
            # Técnicas de injeção
            'has_union', 'has_comment', 'has_semicolon', 
            'has_always_true', 'has_or_injection',
            # Time-based attacks
            'has_time_delay',
            # System exploration
            'has_system_tables',
            # Outras técnicas
            'has_hex_injection', 'has_second_order',
            # Contagens
            'quote_count', 'operator_count',
            # outras
            'has_select_all', 'has_exec', 'has_function_exploit',
            'has_single_quote', 'has_parentheses', 'has_concat_symbols',
            'has_encoding', 'has_union_fragments', 'has_oracle_exploits',
            'has_char_encoding', 'has_load_file_fn', 'has_update', 'has_insert'
        ]
        self.all_feature_names_ordered = []

        self.test_indices = None
        self.y_test_cache = None
        self.y_pred_cache = None
        self.df_cache = None
        self._last_evaluation_data = None
        


    def train(self, df_input: pd.DataFrame):
        """Treina o modelo completo e armazena informações necessárias para predição e análise."""

        start_time = time.time()
        print("[INFO] Iniciando processo de treinamento...")

        try:
            if 'query' not in df_input.columns or 'label' not in df_input.columns:
                raise ValueError("DataFrame de entrada deve conter colunas 'query' e 'label'")

            print(f"[INFO] Total de amostras recebidas: {len(df_input)}")

            df_processed = df_input.copy()
            self.df_cache = df_input.copy()

            print("[INFO] Extraindo features customizadas...")
            df_features_extracted = self.feature_extractor.extract_features(df_processed)
            df_features_extracted = df_features_extracted.dropna(subset=['label'])

            print(f"[INFO] Total de amostras após remoção de 'label' nulo: {len(df_features_extracted)}")

            missing_custom_features = [f for f in self.custom_feature_names if f not in df_features_extracted.columns]
            if missing_custom_features:
                raise ValueError(f"SQLIFeatureExtractor não gerou as seguintes features esperadas: {missing_custom_features}. Features geradas: {df_features_extracted.columns.tolist()}")

            print("[INFO] Gerando matriz TF-IDF...")
            queries_for_tfidf = df_features_extracted['query'].astype(str)
            X_tfidf_matrix = self.vectorizer.fit_transform(queries_for_tfidf)

            print(f"[INFO] TF-IDF gerado com {X_tfidf_matrix.shape[1]} features.")

            self.tfidf_feature_names = self.vectorizer.get_feature_names_out().tolist()
            print(f"[INFO] Primeiras 10 features TF-IDF: {self.tfidf_feature_names[:10]}")

            X_tfidf_df = pd.DataFrame(X_tfidf_matrix.toarray(), columns=self.tfidf_feature_names, index=df_features_extracted.index)
            custom_features_data_unscaled = df_features_extracted[self.custom_feature_names]

            print("[INFO] Concatenando features TF-IDF com features customizadas...")
            X_combined_unscaled = pd.concat([X_tfidf_df, custom_features_data_unscaled], axis=1)
            self.all_feature_names_ordered = X_combined_unscaled.columns.tolist()

            print(f"[INFO] Total de features combinadas (TF-IDF + custom): {len(self.all_feature_names_ordered)}")

            scaled_X_combined_array = self.scaler.fit_transform(X_combined_unscaled)
            X_combined = pd.DataFrame(scaled_X_combined_array, columns=self.all_feature_names_ordered, index=X_combined_unscaled.index)

            X_combined = X_combined.fillna(0)

            y = df_features_extracted['label']
            print(f"[INFO] Distribuição das classes:\n{y.value_counts().to_dict()}")

            class_counts = y.value_counts()
            classes_to_keep = class_counts[class_counts > 1].index
            df_features_extracted = df_features_extracted[df_features_extracted['label'].isin(classes_to_keep)]
            y = df_features_extracted['label']
            X_combined = X_combined.loc[df_features_extracted.index]

            print("[INFO] Dividindo dados em treino e teste...")
            X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.3, random_state=42, stratify=y)
            print(f"[INFO] Tamanho do treino: {len(X_train)}, teste: {len(X_test)}")

            self.test_indices = X_test.index.tolist()

            print("[INFO] Treinando o modelo SVM...")
            self.model.fit(X_train, y_train)

            print("\n[INFO] Calculando importância das features (SVM)...")

            # Inicializa como None para o caso de falha
            feature_importance_data = None

            try:
                # 1. Verifica se o modelo tem coeficientes (SVM linear)
                if hasattr(self.model, 'coef_'):
                    # Obtém os coeficientes (usamos valor absoluto)
                    coef = np.abs(self.model.coef_[0])
                    
                    # Cria DataFrame com as features mais importantes
                    feature_importance = pd.DataFrame({
                        'feature': self.all_feature_names_ordered,
                        'importance': coef
                    }).sort_values('importance', ascending=False)
                    
                    print("\nTop 20 features por coeficiente SVM:")
                    print(feature_importance.head(20))
                    
                    # Prepara os dados para retorno
                    feature_importance_data = {
                        'top_coef_features': feature_importance.head(20).to_dict('records'),
                        'stats': {
                            'mean_importance': coef.mean(),
                            'max_importance': coef.max()
                        }
                    }
                    
                else:
                    print("[AVISO] Este modelo SVM não possui coeficientes (kernel não-linear?)")

            except Exception as e:
                print(f"[ERRO] Falha ao calcular feature importance: {str(e)}")

            print("[INFO] Avaliando modelo...")
            y_pred_test = self.model.predict(X_test)

            self._is_trained = True
            df_original_test_set = self.df_cache.loc[self.test_indices].copy()
            self._last_evaluation_data = (df_original_test_set, y_test.values, y_pred_test, self.test_indices)

            accuracy = accuracy_score(y_test, y_pred_test)
            precision = precision_score(y_test, y_pred_test, average='macro', zero_division=0)
            recall = recall_score(y_test, y_pred_test, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_pred_test, average='macro', zero_division=0)
            report_dict = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)
            conf_matrix_list = confusion_matrix(y_test, y_pred_test).tolist()
            duration = time.time() - start_time

            print(f"[INFO] Treinamento finalizado em {duration:.2f} segundos.")
            print(f"[INFO] Acurácia: {accuracy:.4f}, Precisão: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            print(f"[INFO] Matriz de Confusão: {conf_matrix_list}")

            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "classification_report_dict": report_dict,
                "confusion_matrix_list": conf_matrix_list,
                "training_duration_seconds": duration,
                "model_params": self.model_params,
                "feature_importance": feature_importance_data
            }
        
        except Exception as e:

            self._is_trained = False
            print(f"Erro crítico durante o treinamento: {str(e)}")
            traceback.print_exc()
            return {
                "accuracy": None, "precision": None, "recall": None, "f1_score": None,
                "classification_report_dict": None, "confusion_matrix_list": None, "error": str(e),
                "training_duration_seconds": None, "model_params": self.model_params
            }




    def _prepare_single_query_features(self, query_string: str, return_unscaled: bool = False) -> pd.DataFrame:
        """Prepara as features para uma única query, com opção de retornar versão não escalada."""
        if not self._is_trained:
            raise RuntimeError("O modelo deve ser treinado antes de fazer predições.")

        # DataFrame de entrada
        single_query_df_input = pd.DataFrame({'query': [query_string]})

        # Extrair features customizadas (não escaladas)
        df_features_extracted_single = self.feature_extractor.extract_features(single_query_df_input.copy())
        custom_features_single_data_unscaled = df_features_extracted_single[self.custom_feature_names]

        # Features TF-IDF (não escaladas)
        X_tfidf_matrix_single = self.vectorizer.transform([query_string])
        X_tfidf_df_single = pd.DataFrame(
            X_tfidf_matrix_single.toarray(), 
            columns=self.tfidf_feature_names, 
            index=custom_features_single_data_unscaled.index
        )

        # Combinar features (não escaladas)
        X_combined_unscaled_single = pd.concat([X_tfidf_df_single, custom_features_single_data_unscaled], axis=1)
        X_combined_unscaled_single = X_combined_unscaled_single[self.all_feature_names_ordered]

        # Escalar features (para predição)
        scaled_X_combined_single_array = self.scaler.transform(X_combined_unscaled_single)
        scaled_X_combined_single_df = pd.DataFrame(
            scaled_X_combined_single_array, 
            columns=self.all_feature_names_ordered, 
            index=X_combined_unscaled_single.index
        )

        # Retornar de acordo com o parâmetro
        if return_unscaled:
            return scaled_X_combined_single_df, X_combined_unscaled_single
        return scaled_X_combined_single_df



    def predict_single(self, query_string: str):
        if not self._is_trained:
            raise RuntimeError("Modelo não treinado. Por favor, treine o modelo primeiro.")

        # Obter features escaladas E não escaladas em uma única chamada
        scaled_features_df, unscaled_features_df = self._prepare_single_query_features(query_string, return_unscaled=True)

        # Fazer predição
        prediction = self.model.predict(scaled_features_df)
        probability = self.model.predict_proba(scaled_features_df)

        unscaled_features = unscaled_features_df.iloc[0].to_dict()
        active_features = {
            feature: value 
            for feature, value in unscaled_features.items()
            if feature in self.custom_feature_names and  # Apenas features custom
            ((isinstance(value, (int, float)) and value != 0) or 
                (isinstance(value, bool) and value))
        }

        return {
            "query": query_string,
            "is_malicious": bool(prediction[0] == '1'),
            "probability_benign": probability[0][0],
            "probability_malicious": probability[0][1],
            "label": int(prediction[0]),
            "active_features": active_features  # Somente features ativadas
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