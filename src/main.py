import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Importações internas
from data.data_loader import DataLoader
from features.sqli_features import SQLFeatureExtractor
from models.sqli_detector import SQLInjectionDetector
from evaluation.evaluator import ModelEvaluator
from utils.config import load_config

def run_pipeline(config_path: str, data_path: str, output_dir: str):
    """
    Executa o pipeline completo de detecção de SQL injection.
    
    Args:
        config_path: Caminho para o arquivo de configuração
        data_path: Caminho para os dados de entrada
        output_dir: Diretório para salvar resultados
    """
    try:
        logger.info("Iniciando pipeline de detecção de SQL injection")
        
        # 1. Carregar configurações
        config = load_config(config_path)
        logger.info("Configurações carregadas")
        
        # 2. Carregar dados
        data_loader = DataLoader()
        df = data_loader.load_csv(data_path)
        logger.info(f"Dados carregados. Total de registros: {len(df)}")
        
        # 3. Extração de features
        feature_extractor = SQLFeatureExtractor(config['features'])
        features = feature_extractor.extract(df)
        logger.info("Features extraídas com sucesso")
        
        # 4. Treinamento do modelo
        model = SQLInjectionDetector(config['model'])
        model.train(features, df['label'])
        logger.info("Modelo treinado com sucesso")
        
        # 5. Avaliação
        evaluator = ModelEvaluator()
        report = evaluator.evaluate(model, features, df['label'])
        
        # 6. Salvar resultados
        save_results(report, output_dir)
        logger.info(f"Resultados salvos em {output_dir}")
        
    except Exception as e:
        logger.error(f"Erro durante a execução do pipeline: {str(e)}")
        raise

def save_results(report: dict, output_dir: str):
    """Salva os resultados da avaliação"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Salva relatório em texto
    with open(output_path / 'report.txt', 'w') as f:
        f.write(str(report))
    
    # Aqui você pode adicionar lógica para salvar:
    # - O modelo treinado
    # - Visualizações
    # - Arquivos de configuração usados

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='SQL Injection Detection Pipeline')
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/config.yaml',
        help='Caminho para o arquivo de configuração'
    )
    parser.add_argument(
        '--data', 
        type=str, 
        required=True,
        help='Caminho para o arquivo de dados CSV'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='results',
        help='Diretório de saída para os resultados'
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        config_path=args.config,
        data_path=args.data,
        output_dir=args.output
    )