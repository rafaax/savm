import logging
import joblib
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any, Union, Optional, List
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline as make_imb_pipeline
from features.sqli_features import SQLIFeatureExtractor
from features.text_features import TextFeatureExtractor
from evaluation.evaluator import ModelEvaluator
from evaluation.visualizer import ModelVisualizer

class ModelManager:
    """
    Gerencia o ciclo de vida completo de modelos de detecção de SQL Injection.
    
    Funcionalidades:
    - Pré-processamento automático
    - Balanceamento de classes
    - Treinamento de múltiplos modelos
    - Validação cruzada
    - Otimização de hiperparâmetros
    - Serialização de modelos
    """
    
    def __init__(self, config: Dict[str, Any], models_dir: Union[str, Path] = "models"):
        """
        Inicializa o gerenciador com configuração.
        
        Args:
            config: Dicionário de configuração
            models_dir: Diretório para salvar/ler modelos
        """
        self.config = config
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Inicializa componentes
        self.feature_extractor = self._init_feature_extractor()
        self.evaluator = ModelEvaluator(class_names={0: "Normal", 1: "SQLi"})
        self.visualizer = ModelVisualizer()
        
        self.current_model = None
        self.best_model = None

    def _init_feature_extractor(self) -> Pipeline:
        """Configura o pipeline de extração de features."""
        return Pipeline([
            ('sqli_features', SQLIFeatureExtractor(self.config.get('sqli_features'))),
            ('text_features', TextFeatureExtractor(self.config.get('text_features')))
        ])

    def load_data(self, data_path: Union[str, Path], test_size: float = 0.2) -> Dict[str, pd.DataFrame]:
        """
        Carrega e prepara os dados para treinamento.
        
        Args:
            data_path: Caminho para o arquivo de dados
            test_size: Proporção para o conjunto de teste
            
        Returns:
            Dicionário com DataFrames processados
        """
        try:
            df = pd.read_csv(data_path)
            df = df.dropna(subset=['query'])
            
            # Extração de features
            X = self.feature_extractor.fit_transform(df)
            y = df['label']
            
            # Divisão treino-teste
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                stratify=y,
                random_state=self.config.get('random_state', 42)
            )
            
            self.logger.info(f"Dados carregados. Treino: {len(X_train)}, Teste: {len(X_test)}")
            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar dados: {str(e)}")
            raise

    def train_model(self, model, data: Dict[str, pd.DataFrame], model_name: str) -> Dict[str, Any]:
        """
        Treina um modelo com os dados fornecidos.
        
        Args:
            model: Instância do modelo scikit-learn
            data: Dados de treino/teste
            model_name: Nome para identificação do modelo
            
        Returns:
            Resultados do treinamento
        """
        try:
            self.logger.info(f"Iniciando treinamento do modelo {model_name}")
            
            # Pipeline com balanceamento
            pipeline = self._build_pipeline(model)
            
            # Treinamento
            pipeline.fit(data['X_train'], data['y_train'])
            
            # Avaliação
            train_pred = pipeline.predict(data['X_train'])
            test_pred = pipeline.predict(data['X_test'])
            
            # Métricas
            results = {
                'model': pipeline,
                'model_name': model_name,
                'train_report': classification_report(data['y_train'], train_pred, output_dict=True),
                'test_report': classification_report(data['y_test'], test_pred, output_dict=True),
                'feature_importances': self._get_feature_importance(pipeline)
            }
            
            self.current_model = results
            if not self.best_model or results['test_report']['weighted avg']['f1-score'] > self.best_model['test_report']['weighted avg']['f1-score']:
                self.best_model = results
                self.logger.info(f"Novo melhor modelo: {model_name}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Erro no treinamento: {str(e)}")
            raise

    def _build_pipeline(self, model) -> Pipeline:
        """Constroi pipeline com técnicas de balanceamento."""
        sampling_strategy = self.config.get('sampling', 'smote')
        
        if sampling_strategy == 'smote':
            sampler = SMOTE(
                sampling_strategy='auto',
                random_state=self.config.get('random_state', 42),
                k_neighbors=5
            )
        elif sampling_strategy == 'undersample':
            sampler = RandomUnderSampler(
                sampling_strategy='majority',
                random_state=self.config.get('random_state', 42)
            )
        else:
            return model
            
        return make_imb_pipeline(sampler, model)

    def _get_feature_importance(self, pipeline) -> Optional[Dict[str, float]]:
        """Extrai importância das features do modelo."""
        try:
            # Tenta extrair importâncias para vários tipos de modelos
            if hasattr(pipeline.steps[-1][1], 'feature_importances_'):
                importances = pipeline.steps[-1][1].feature_importances_
            elif hasattr(pipeline.steps[-1][1], 'coef_'):
                importances = pipeline.steps[-1][1].coef_[0]
            else:
                return None
                
            feature_names = self.feature_extractor.get_feature_names_out()
            return dict(zip(feature_names, importances))
        except Exception:
            return None

    def evaluate_model(self, model_results: Dict[str, Any], data: Dict[str, pd.DataFrame], output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Avalia um modelo e gera relatórios.
        
        Args:
            model_results: Resultados do treinamento
            data: Dados de teste
            output_dir: Diretório para salvar resultados
            
        Returns:
            Resultados completos da avaliação
        """
        try:
            evaluation = self.evaluator.evaluate_model(
                model_results['model'],
                data['X_test'],
                data['y_test'],
                data['X_train'],
                data['y_train']
            )
            
            if output_dir:
                saved_files = self.visualizer.generate_report(
                    evaluation,
                    output_dir=output_dir,
                    model_name=model_results['model_name']
                )
                evaluation['saved_files'] = saved_files
            
            model_results.update({'evaluation': evaluation})
            return model_results
            
        except Exception as e:
            self.logger.error(f"Erro na avaliação: {str(e)}")
            raise

    def save_model(self, model_results: Dict[str, Any], save_dir: Optional[str] = None) -> Path:
        """
        Salva o modelo e seus artefatos em disco.
        
        Args:
            model_results: Resultados do modelo a serem salvos
            save_dir: Subdiretório para salvar (opcional)
            
        Returns:
            Caminho onde o modelo foi salvo
        """
        try:
            model_name = model_results['model_name']
            save_path = self.models_dir / save_dir if save_dir else self.models_dir
            save_path = save_path / model_name
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Salva o modelo
            joblib.dump(model_results['model'], save_path / 'model.joblib')
            
            # Salva metadados
            metadata = {
                'model_name': model_name,
                'training_date': pd.Timestamp.now().isoformat(),
                'performance': {
                    'train_f1': model_results['train_report']['weighted avg']['f1-score'],
                    'test_f1': model_results['test_report']['weighted avg']['f1-score']
                },
                'feature_list': list(self.feature_extractor.get_feature_names_out())
            }
            
            import json
            with open(save_path / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Modelo {model_name} salvo em {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar modelo: {str(e)}")
            raise

    def load_model(self, model_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Carrega um modelo salvo.
        
        Args:
            model_path: Caminho para o diretório do modelo
            
        Returns:
            Modelo e metadados carregados
        """
        try:
            model_path = Path(model_path)
            
            model = joblib.load(model_path / 'model.joblib')
            
            with open(model_path / 'metadata.json') as f:
                metadata = json.load(f)
            
            return {
                'model': model,
                'metadata': metadata
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo: {str(e)}")
            raise

    def get_feature_names(self) -> List[str]:
        """Retorna a lista de features usadas no modelo."""
        return self.feature_extractor.get_feature_names_out()