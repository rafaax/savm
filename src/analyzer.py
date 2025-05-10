import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import pandas as pd

class ResultAnalyzer:
    @staticmethod
    def analyze_false_negatives(df, y_test, y_pred, test_indices):
        """Analisa falsos negativos detalhadamente"""
        # Cria uma série temporária com os resultados para os índices de teste
        temp_results = pd.DataFrame({
            'y_test': y_test,
            'y_pred': y_pred
        }, index=test_indices)
        
        # Filtra o DataFrame original usando os índices dos falsos negativos
        fn = df.loc[temp_results[(temp_results['y_test'] == 1) & 
                                (temp_results['y_pred'] == 0)].index].copy()
        
        if len(fn) > 0:
            # Estatísticas básicas
            fn['length'] = fn['query'].str.len()
            print("\nEstatísticas de Falsos Negativos:")
            print(f"Total: {len(fn)}")
            print(fn['length'].describe())
            
            # Padrões mais comuns
            common_patterns = Counter(fn['query'].str[:30])
            print("\nPadrões mais frequentes:")
            for pattern, count in common_patterns.most_common(10):
                print(f"{count}x → {pattern!r}")
            
            # Visualização
            plt.figure(figsize=(12, 6))
            fn['length'].hist(bins=20, color='coral', edgecolor='black')
            plt.title('Distribuição de Tamanhos dos Falsos Negativos', pad=20)
            plt.xlabel('Número de Caracteres')
            plt.ylabel('Quantidade')
            plt.show()
            
            return fn
        else:
            print("Nenhum falso negativo detectado!")
            return None
