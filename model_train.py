import pandas as pd
from src.sqli_detector import SQLIDetector # Importa a classe atualizada
from src.analyzer import ResultAnalyzer # Supondo que ResultAnalyzer está atualizado
import os

# Define o caminho onde o modelo será salvo
MODEL_SAVE_PATH = 'models/sqli_detector_model.joblib'
DATASET_PATH = 'mocks/dataset.csv'

def main():

    try:
        df = pd.read_csv(DATASET_PATH)
        print(f"Dados carregados de {DATASET_PATH}. Total: {len(df)} registros")
        print(f"Distribuição de classes:\n{df['label'].value_counts(normalize=True)}")
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return
    
    detector = SQLIDetector()
    print("\nIniciando treinamento do modelo SQLIDetector...")
    
    # O método train agora retorna um dicionário de métricas.
    # O detector armazena y_test, y_pred, test_indices internamente.
    training_metrics = detector.train(df)
    
    if not detector.is_trained() or not training_metrics:
        print("Falha no treinamento do modelo. Abortando.")
        return

    print("\nMétricas de Treinamento:")
    print(f"  Acurácia: {training_metrics.get('accuracy', 'N/A'):.4f}")
    # Você pode imprimir mais detalhes do classification_report se desejar

    # Salva o modelo treinado
    print(f"\nSalvando o modelo treinado em {MODEL_SAVE_PATH}...")
    detector.save_model(MODEL_SAVE_PATH) # Chama o novo método

    # Análise detalhada usando os dados cacheados pelo detector
    print("\nIniciando análise de falsos negativos...")
    evaluation_data = detector.get_last_evaluation_data()
    
    if evaluation_data:
        df_original_for_analysis, y_test_cached, y_pred_cached, test_indices_cached = evaluation_data
        
        analyzer = ResultAnalyzer() # Supondo que ResultAnalyzer está definido e adaptado
        # O ResultAnalyzer.analyze_false_negatives foi adaptado para retornar (fn_df, analysis_details)
        fn_df, analysis_details = analyzer.analyze_false_negatives(
            df_original_for_analysis, 
            y_test_cached, 
            y_pred_cached, 
            test_indices_cached
        )
        
        if fn_df is not None and not fn_df.empty:
            os.makedirs('results', exist_ok=True) # Garante que o diretório results exista
            output_csv_path = 'results/false_negatives_analysis_from_train_script.csv'
            fn_df.to_csv(output_csv_path, index=False)
            print(f"\nFalsos negativos ({len(fn_df)}) salvos em '{output_csv_path}'")
            if analysis_details:
                print("Detalhes da Análise de Falsos Negativos:")
                if 'length_statistics' in analysis_details:
                    print(f"  Estatísticas de Comprimento (Exemplo - Média): {analysis_details['length_statistics'].get('mean', 'N/A')}")
                if 'common_patterns' in analysis_details and analysis_details['common_patterns']:
                    print(f"  Padrão Mais Comum: {analysis_details['common_patterns'][0]}")
        else:
            print("\nNenhum falso negativo encontrado ou análise não pôde ser realizada.")
    else:
        print("\nNenhum dado de avaliação disponível no detector para análise de falsos negativos.")

if __name__ == "__main__":
    main()