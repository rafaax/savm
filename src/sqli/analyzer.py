from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from sklearn.metrics import recall_score

class ResultAnalyzer:
    @staticmethod
    def save_feature_importance_plot(importance_df, filename, top_n=20):
        """
        Salva um gráfico de barras horizontais com as features mais importantes.
        
        """
        try:
            plt.figure(figsize=(12, 8))
            
            # Seleciona e ordena as top_n features
            top_features = importance_df.sort_values('importance_mean', ascending=True).tail(top_n)
            
            # Cria o gráfico de barras horizontais
            bars = plt.barh(
                top_features['feature'], 
                top_features['importance_mean'],
                xerr=top_features['importance_std'],
                color='skyblue',
                edgecolor='black'
            )
            
            # Adiciona rótulos e título
            plt.title(f'Top {top_n} Features by Importance', pad=20)
            plt.xlabel('Importance Score')
            plt.ylabel('Features')
            
            # Adiciona os valores nas barras
            for bar in bars:
                width = bar.get_width()
                plt.text(width, bar.get_y() + bar.get_height()/2, 
                         f'{width:.3f}', 
                         va='center', ha='left', fontsize=9)
            
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            # Cria o diretório se não existir
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Feature importance plot saved to: {filename}")
            return True
        
        except Exception as e:
            print(f"Error saving feature importance plot: {e}")
            return False
        
    @staticmethod
    def analyze_false_negatives(df, y_test, y_pred, test_indices, results_dir: str, timestamp: str):
        """
        Analisa instâncias classificadas como falsos negativos para identificar padrões e características.
        """
        
        analysis_details = {
            "false_negatives_count": 0,
            "length_statistics": None,
            "common_patterns_prefix": [],
            "keyword_patterns": {},
            "feature_differences": {},
            "overall_recall_on_test": None
        }

        # --- Validação de Entrada ---
        if test_indices is None or y_test is None or y_pred is None or df is None:
            print("Analyzer: Dados de entrada inválidos (None).")
            return pd.DataFrame(), analysis_details

        if len(y_test) != len(y_pred):
            print(f"Analyzer ERRO: y_test ({len(y_test)}) e y_pred ({len(y_pred)}) têm comprimentos diferentes.")
            return pd.DataFrame(), analysis_details

        if len(test_indices) != len(y_test):
            print(f"Analyzer ERRO: test_indices ({len(test_indices)}) e y_test ({len(y_test)}) têm comprimentos diferentes.")
            if len(test_indices) == len(df): # Se test_indices parece ser o índice completo
                print("Analyzer AVISO: test_indices parece ser o índice completo do df. Usando-o.")
            else:
                print("Analyzer ERRO: Inconsistência nos comprimentos de test_indices, y_test/y_pred.")
                return pd.DataFrame(), analysis_details


        # --- Alinhar y_test e y_pred com os índices originais ---
        try:
            y_test_series = pd.Series(y_test, index=test_indices, name='y_test')
            y_pred_series = pd.Series(y_pred, index=test_indices, name='y_pred')

            valid_indices = y_test_series.index.intersection(df.index)
            if len(valid_indices) != len(y_test_series):
                print(f"Analyzer AVISO: Nem todos os test_indices encontrados no DataFrame original. Usando {len(valid_indices)}/{len(y_test_series)} índices válidos.")
                y_test_series = y_test_series.loc[valid_indices]
                y_pred_series = y_pred_series.loc[valid_indices]
                # Atualizar y_test e y_pred para corresponder aos índices válidos, se necessário para cálculos futuros
                y_test = y_test_series.values
                y_pred = y_pred_series.values


            temp_results = pd.DataFrame({'y_test': y_test_series, 'y_pred': y_pred_series})

        except Exception as e:
            print(f"Analyzer ERRO ao criar temp_results ou alinhar índices: {e}")
            return pd.DataFrame(), analysis_details

        # --- Identificar Falsos Negativos ---
        false_negatives_mask = (temp_results['y_test'] == 1) & (temp_results['y_pred'] == 0)
        fn_indices = temp_results[false_negatives_mask].index
        fn_df = df.loc[fn_indices].copy() if not fn_indices.empty else pd.DataFrame()

        analysis_details['false_negatives_count'] = len(fn_df)

        # --- Análises Adicionais ---
        if not fn_df.empty:
            print(f"Analyzer: Encontrados {len(fn_df)} falsos negativos.")

            # 1. Análise de Comprimento
            fn_df['length'] = fn_df['query'].astype(str).apply(len)
            length_desc = fn_df['length'].describe()
            analysis_details['length_statistics'] = length_desc.to_dict()

            # 2. Análise de Padrões Comuns (Prefixos)  Útil para ver se FNs começam de forma similar
            common_patterns_counter = Counter(fn_df['query'].astype(str).str[:30])
            patterns_for_details = []
            for pattern, count in common_patterns_counter.most_common(10):
                patterns_for_details.append({"pattern": pattern, "count": count})
            analysis_details['common_patterns_prefix'] = patterns_for_details

            # 3. Análise de Padrões de Palavras-chave (SQL) Focar em palavras-chave comuns em SQL Injection
            keywords = ['SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE', 'DELETE', 'OR', 'AND', 'UNION', 'SLEEP', 'BENCHMARK'] # Adicionado mais keywords relevantes
            keyword_counter = Counter() # Usar regex para encontrar palavras inteiras e case-insensitive
            for query in fn_df['query'].astype(str):
                for keyword in keywords:
                    # Procura pela palavra inteira, case-insensitive
                    if pd.Series(query).str.contains(r'\b' + keyword + r'\b', case=False, regex=True).any():
                        keyword_counter[keyword] += 1

            analysis_details['keyword_patterns'] = dict(keyword_counter.most_common()) # Salva os mais comuns primeiro

            # 4. Análise de Diferenças em Características
            feature_columns = [col for col in df.columns if col not in ['query', 'label', 'y_test', 'y_pred', 'length']] # Exclui colunas não-características
            if feature_columns:
                 print(f"Analyzer: Analisando diferenças nas características: {feature_columns}")
                 try:
                    # Estatísticas descritivas para features numéricas nos FNs
                    fn_numeric_features = fn_df[feature_columns].select_dtypes(include=['number'])
                    if not fn_numeric_features.empty:
                        feature_desc = fn_numeric_features.describe()
                        analysis_details['feature_differences']['numeric'] = feature_desc.to_dict()

                    # Contagem para features categóricas nos FNs
                    fn_categorical_features = fn_df[feature_columns].select_dtypes(exclude=['number', 'object']) # Considerando dtypes como category ou bool
                    if not fn_categorical_features.empty:
                        categorical_counts = {}
                        for col in fn_categorical_features.columns:
                            categorical_counts[col] = fn_categorical_features[col].value_counts().to_dict()
                        analysis_details['feature_differences']['categorical'] = categorical_counts

                    # Análise para features textuais/object (pode ser mais complexo, talvez amostra)
                    fn_object_features = fn_df[feature_columns].select_dtypes(include=['object'])
                    if not fn_object_features.empty:
                        object_sample = {}
                        for col in fn_object_features.columns:
                            # Pega os 5 valores mais comuns
                            object_sample[col] = fn_object_features[col].value_counts().head(5).to_dict()
                        analysis_details['feature_differences']['object_sample'] = object_sample


                 except Exception as e:
                     print(f"Analyzer AVISO: Erro ao analisar diferenças de características: {e}")
                     analysis_details['feature_differences']['error'] = str(e)
            else:
                 print("Analyzer AVISO: Nenhuma coluna de característica identificada para análise de diferenças.")


            # 5. Cálculo de Métricas Focadas (Recall Geral) O Recall é a métrica mais impactada pelos Falsos Negativos
            try:
                if (y_test == 1).any():
                    overall_recall = recall_score(y_test, y_pred, zero_division=0)
                    analysis_details['overall_recall_on_test'] = overall_recall
                    print(f"Analyzer: Recall geral no conjunto de teste: {overall_recall:.4f}")
                else:
                    print("Analyzer AVISO: Nenhum rótulo positivo no y_test para calcular Recall.")
                    analysis_details['overall_recall_on_test'] = "N/A (No positive labels in y_test)"

            except Exception as e:
                print(f"Analyzer AVISO: Erro ao calcular Recall: {e}")
                analysis_details['overall_recall_on_test'] = f"Error: {e}"


            # --- Visualização (Histograma de Comprimento dos FNs) ---
            try:
                plt.figure(figsize=(12, 6))
                fn_df['length'].hist(bins=50, color='coral', edgecolor='black') # Aumentado bins para mais detalhe
                plt.title(f'Distribuição de Tamanhos dos Falsos Negativos ({timestamp})', pad=20)
                plt.xlabel('Número de Caracteres')
                plt.ylabel('Quantidade')
                plt.grid(axis='y', alpha=0.75) # Adiciona grid
                
                os.makedirs(results_dir, exist_ok=True)
                plot_filename = f"fn_length_distribution_{timestamp}.png"
                plot_save_path = os.path.join(results_dir, plot_filename)
                plt.savefig(plot_save_path, bbox_inches='tight') # bbox_inches para evitar cortar rótulos
                plt.close()
                print(f"Analyzer: Histograma salvo em {plot_save_path}")

            except Exception as e:
                print(f"Analyzer AVISO: Erro ao gerar/salvar plotagem do histograma: {e}")

            # --- Salvar Detalhes da Análise ---
            try:
                analysis_filename = f"fn_analysis_details_{timestamp}.json"
                analysis_save_path = os.path.join(results_dir, analysis_filename)
                with open(analysis_save_path, 'w', encoding='utf-8') as f:
                    json.dump(analysis_details, f, indent=4, ensure_ascii=False)
                print(f"Analyzer: Detalhes da análise salvos em {analysis_save_path}")
            except Exception as e:
                print(f"Analyzer AVISO: Erro ao salvar detalhes da análise em JSON: {e}")


            return fn_df, analysis_details
        else:
            print("Analyzer: Nenhum falso negativo detectado.")
            return pd.DataFrame(), analysis_details
