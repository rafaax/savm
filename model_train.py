import pandas as pd
from src.sqli_detector import SQLIDetector
from src.analyzer import ResultAnalyzer
import os
from datetime import datetime
from database import SessionLocal, TrainedModelLog
from pathlib import Path 

MODELS_DIR = 'models'
DATASET_PATH = 'mocks/dataset.csv'
RESULTS_DIR = 'results'

def main():

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)

    try:
        df = pd.read_csv(DATASET_PATH)
        print(f"Dados carregados de {DATASET_PATH}. Total: {len(df)} registros")
        print(f"Distribuição de classes:\n{df['label'].value_counts(normalize=True)}")
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return
    
    detector = SQLIDetector()
    print("\nIniciando treinamento do modelo SQLIDetector...")
    
    training_metrics = detector.train(df)
    
    if not detector.is_trained() or not training_metrics:
        print("Falha no treinamento do modelo. Abortando.")
        return
    
    current_accuracy = training_metrics.get('accuracy')
    current_precision = training_metrics.get('precision')
    current_recall = training_metrics.get('recall')
    current_f1_score = training_metrics.get('f1_score')

    print("\nMétricas de Treinamento:")
    print(f"  Acurácia: {current_accuracy:.4f}" if current_accuracy is not None else "  Acurácia: N/A")
    print(f"  Precisão: {current_precision:.4f}" if current_precision is not None else "  Precisão: N/A")
    print(f"  Recall:   {current_recall:.4f}" if current_recall is not None else "  Recall: N/A")
    print(f"  F1-Score: {current_f1_score:.4f}" if current_f1_score is not None else "  F1-Score: N/A")

    # --- GERAÇÃO DINÂMICA DO NOME DO ARQUIVO DO MODELO ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"sqli_detector_model_{timestamp}.joblib"
    current_model_save_path = os.path.join(MODELS_DIR, model_filename)
    
    print(f"\nSalvando o modelo treinado em {current_model_save_path}...")
    detector.save_model(current_model_save_path)

    latest_model_info_file = os.path.join(MODELS_DIR, "latest_model_info.txt")
    with open(latest_model_info_file, "w") as f:
        f.write(model_filename)
    print(f"Informação do último modelo ('{model_filename}') salva em '{latest_model_info_file}'")


    
    print("\nIniciando análise de falsos negativos...")
    evaluation_data = detector.get_last_evaluation_data() #  Análise detalhada usando os dados cacheados pelo detector
    
    if evaluation_data:
        df_original_for_analysis, y_test_cached, y_pred_cached, test_indices_cached = evaluation_data
        analyzer = ResultAnalyzer()
        fn_df, analysis_details = analyzer.analyze_false_negatives(
            df_original_for_analysis, 
            y_test_cached, 
            y_pred_cached, 
            test_indices_cached,
            RESULTS_DIR,
            timestamp
        )
        
        if fn_df is not None and not fn_df.empty:
            # Salva análise com timestamp também para corresponder ao modelo
            fn_analysis_filename = f"false_negatives_analysis_{timestamp}.csv"
            output_csv_path = os.path.join(RESULTS_DIR, fn_analysis_filename)
            fn_df.to_csv(output_csv_path, index=False)
            print(f"\nFalsos negativos ({len(fn_df)}) salvos em '{output_csv_path}'")
            
            if analysis_details:
                print("Detalhes da Análise de Falsos Negativos:")
                # ... (impressão dos detalhes da análise)
        else:
            print("\nNenhum falso negativo encontrado ou análise não pôde ser realizada.")
    else:
        print("\nNenhum dado de avaliação disponível no detector para análise de falsos negativos.")

if __name__ == "__main__":
    main()