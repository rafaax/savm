import pandas as pd
from src.analyzer import ResultAnalyzer
from src.sqli_detector import SQLIDetector


def main():
    try:
        df = pd.read_csv('mocks/dataset.csv')
        print(f"Dados carregados. Total: {len(df)} registros")
        print(f"Distribuição de classes:\n{df['label'].value_counts()}")
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return
    
    # Pipeline completo
    detector = SQLIDetector()
    y_test, y_pred = detector.train(df)
    
    # Análise detalhada usando os índices corretos
    analyzer = ResultAnalyzer()
    fn_df = analyzer.analyze_false_negatives(df, y_test, y_pred, detector.test_indices)
    
    # Salva falsos negativos para análise posterior
    if fn_df is not None:
        fn_df.to_csv('results/false_negatives_analysis.csv', index=False)
        print("\nFalsos negativos salvos em 'results/false_negatives_analysis.csv'")

if __name__ == "__main__":
    main()