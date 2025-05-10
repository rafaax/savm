import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import os

class ResultAnalyzer:
    @staticmethod
    def analyze_false_negatives(df, y_test, y_pred, test_indices, results_dir: str, timestamp: str): # Novos parâmetros
        """
        Analisa falsos negativos detalhadamente, retorna o DataFrame e os detalhes da análise.
        Salva o plot da distribuição de comprimentos.
        """
        
        analysis_details = {
            "length_statistics": None,
            "common_patterns": []
        }

        # Validação básica de entrada
        if test_indices is None or y_test is None or y_pred is None or df is None:
            print("Analyzer: Dados de entrada inválidos (None) para análise de FNs.")
            return pd.DataFrame(), analysis_details # Retorna DF vazio e detalhes vazios

        if len(y_test) != len(y_pred):
            print("Analyzer ERRO: y_test e y_pred têm comprimentos diferentes.")
            return pd.DataFrame(), analysis_details

        try:
            # Se y_test já é uma Series do Pandas com os test_indices corretos
            if isinstance(y_test, pd.Series) and y_test.index.equals(pd.Index(test_indices)):
                y_pred_series = pd.Series(y_pred, index=test_indices) # y_test já está correto
            else: # Tenta construir Series com os test_indices
                y_test_series = pd.Series(y_test, index=test_indices)
                y_pred_series = pd.Series(y_pred, index=test_indices)
                y_test = y_test_series # Atualiza y_test para ser a Series com índice

            temp_results = pd.DataFrame({
                'y_test': y_test, # y_test agora é uma Series com o índice correto
                'y_pred': y_pred_series
            })
        except Exception as e:
            print(f"Analyzer ERRO ao criar temp_results para análise de FNs: {e}")
            print(f"  len(y_test): {len(y_test)}, len(y_pred): {len(y_pred)}, len(test_indices): {len(test_indices)}")
            return pd.DataFrame(), analysis_details

        
        false_negatives_mask = (temp_results['y_test'] == 1) & (temp_results['y_pred'] == 0)
        fn_indices = temp_results[false_negatives_mask].index
        fn_df = df.loc[fn_indices].copy() if not fn_indices.empty else pd.DataFrame()
        

        if not fn_df.empty:
            fn_df['length'] = fn_df['query'].astype(str).apply(len)
            print("\nEstatísticas de Falsos Negativos:")
            print(f"Total: {len(fn_df)}")
            length_desc = fn_df['length'].describe()
            print(length_desc)
            analysis_details['length_statistics'] = length_desc.to_dict()
            
            common_patterns_counter = Counter(fn_df['query'].astype(str).str[:30]) # Primeiros 30 chars
            print("\nPadrões mais frequentes (início da query):")
            patterns_for_details = []
            for pattern, count in common_patterns_counter.most_common(10):
                print(f"{count}x → {pattern!r}")
                patterns_for_details.append({"pattern": pattern, "count": count})
            analysis_details['common_patterns'] = patterns_for_details
            
            try:
                plt.figure(figsize=(12, 6))
                fn_df['length'].hist(bins=20, color='coral', edgecolor='black')
                plt.title(f'Distribuição de Tamanhos dos Falsos Negativos ({timestamp})', pad=20)
                plt.xlabel('Número de Caracteres')
                plt.ylabel('Quantidade')
                
                os.makedirs(results_dir, exist_ok=True)
                plot_filename = f"fn_length_distribution_{timestamp}.png"
                plot_save_path = os.path.join(results_dir, plot_filename)
                plt.savefig(plot_save_path)
                plt.close() 

            except Exception as e:
                print(f"AVISO: Erro ao gerar/salvar plotagem no analyzer: {e}")

            return fn_df, analysis_details
        else:
            print("Analyzer: Nenhum falso negativo detectado.")
            return pd.DataFrame(), analysis_details