import pandas as pd
from sqli.sqli_detector import SQLIDetector
from sqli.analyzer import ResultAnalyzer
import os
from datetime import datetime, timezone
from db.db_setup import SessionLocal, TrainedModelLog
from pathlib import Path 
import json

MODELS_DIR = 'models'
DATASET_PATH = 'mocks/dataset-curta+sqliv3.csv'
RESULTS_DIR = 'results'

def main():

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)

    try:
        df = pd.read_csv(DATASET_PATH)
        print(f"Dados carregados do dataset:  {DATASET_PATH}")
        print(f"Quantidade de dados: {len(df)} registros")
        print(f"Distribuição de classes:\n{df['label'].value_counts(normalize=True)}")
    except Exception as e:
        print(f"Erro ao carregar dados do dataset: {e}")
        return
    
    detector = SQLIDetector() # instanciando a classe do sqli detector que faz o treinamento, predição e etc..
    analyzer = ResultAnalyzer() # intancianciando a classe que faz o analisador de graficcos e etc... 
    print("\nIniciando treinamento do modelo") 
    
    training_metrics = detector.train(df)
    
    if not detector.is_trained() or not training_metrics:
        print("Falha no treinamento do modelo :(")
        return
    
    current_accuracy = training_metrics.get('accuracy')
    current_precision = training_metrics.get('precision')
    current_recall = training_metrics.get('recall')
    current_f1_score = training_metrics.get('f1_score')
    training_duration = training_metrics.get('training_duration_seconds')
    model_params = training_metrics.get('model_params')
    feature_importance = training_metrics.get('feature_importance')

    feature_plot_path = None
    perm_importance_csv = None
    
    if feature_importance:
        print("\nProcessando feature importance...")
        feature_analysis_dir = os.path.join(RESULTS_DIR, "feature_analysis")
        os.makedirs(feature_analysis_dir, exist_ok=True)
        
        # Converte para DataFrame
        perm_importance_df = pd.DataFrame(feature_importance.get('top_perm_features', {}))
        
        # Salva CSV
        perm_importance_csv = os.path.join(feature_analysis_dir, f"feature_importance_{timestamp_str}.csv")
        perm_importance_df.to_csv(perm_importance_csv, index=False)
        
        # Gera e salva o gráfico usando o método do analyzer
        feature_plot_path = os.path.join(feature_analysis_dir, f"feature_importance_{timestamp_str}.png")
        analyzer.save_feature_importance_plot(
            importance_df=perm_importance_df,
            filename=feature_plot_path,
            top_n=15  # Mostra as 15 features mais importantes
        )
        
        print(f"  - Feature importance salva em: {perm_importance_csv}")
        print(f"  - Gráfico de features salvo em: {feature_plot_path}")

    print("\nMétricas de Treinamento:")
    print(f"  Acurácia: {current_accuracy:.4f}" if current_accuracy is not None else "  Acurácia: N/A")
    print(f"  Precisão: {current_precision:.4f}" if current_precision is not None else "  Precisão: N/A")
    print(f"  Recall:   {current_recall:.4f}" if current_recall is not None else "  Recall: N/A")
    print(f"  F1-Score: {current_f1_score:.4f}" if current_f1_score is not None else "  F1-Score: N/A")
    print(f"  Duração do Treinamento: {training_duration:.2f} segundos" if training_duration is not None else "  Duração do Treinamento: N/A")

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_filename = f"sqli_detector_model_{timestamp_str}.joblib"
    current_model_save_path = os.path.join(MODELS_DIR, model_filename)
    
    detector.save_model(current_model_save_path)

    latest_model_info_file = os.path.join(MODELS_DIR, "latest_model_info.txt")
    
    try:
        with open(latest_model_info_file, "w") as f:
            f.write(model_filename)
    except Exception as e:
        print(f"ERRO ao salvar informação do último modelo: {e}")


    
    print("\nIniciando análise de falsos negativos...")
    evaluation_data = detector.get_last_evaluation_data() #  Análise detalhada usando os dados cacheados pelo detector

    if evaluation_data:
        df_original_for_analysis, y_test_cached, y_pred_cached, test_indices_cached = evaluation_data

        fn_df, analysis_details = analyzer.analyze_false_negatives(
            df_original_for_analysis, 
            y_test_cached, 
            y_pred_cached, 
            test_indices_cached,
            RESULTS_DIR,
            timestamp_str
        )
        

        fn_count = analysis_details['false_negatives_count']  # Atualizando fn_count após a análise
        
        if not fn_df.empty:
            fn_analysis_filename = f"false_negatives_analysis_{timestamp_str}.csv"
            fn_report_file_path = os.path.join(RESULTS_DIR, fn_analysis_filename)
            fn_df.to_csv(fn_report_file_path, index=False)
        else:
            print("\nNenhum falso negativo encontrado ou análise não pôde ser realizada.")
            fn_report_file_path = None
            
    else:
        print("\nNenhum dado de avaliação disponível no detector para análise de falsos negativos.")



    print("\nRegistrando informações do modelo treinado no banco de dados...")
    db_session = None
    try:
        db_session = SessionLocal()
        
        new_model_log = TrainedModelLog(
            model_filename=model_filename,
            model_path=str(Path(current_model_save_path).resolve()),
            dataset_used_path=str(Path(DATASET_PATH).resolve()),
            accuracy=current_accuracy,
            precision=current_precision,
            recall=current_recall,
            f1_score=current_f1_score,
            training_duration_seconds=training_duration,
            false_negatives_count=fn_count,
            false_negatives_report_path=str(Path(fn_report_file_path).resolve()) if fn_report_file_path else None,
            feature_importance_path=str(Path(perm_importance_csv).resolve()) if feature_importance else None,
            feature_importance_plot_path=str(Path(feature_plot_path).resolve()) if feature_importance else None,
            notes=f"Modelo treinado em {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}.",
            model_params=json.dumps(model_params),
            feature_importance_data=json.dumps(feature_importance) if feature_importance else None
        )

        db_session.add(new_model_log)
        db_session.commit()
        db_session.refresh(new_model_log)
        print(f"Modelo '{new_model_log.model_filename}' salvo no banco de dados.")

    except Exception as e:
        print(f"ERRO AO REGISTRAR AS INFORMAÇÕES DO MODELO TREINADO NO DB: {e}")
        if db_session:
            db_session.rollback()
    finally:
        if db_session:
            db_session.close()

if __name__ == "__main__":
    main()