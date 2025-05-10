import logging
from typing import Dict, Tuple, Any, Optional, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
    average_precision_score,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import learning_curve
from pathlib import Path

class ModelEvaluator:
    """
    Classe para avaliação abrangente de modelos de detecção de SQL injection.
    
    Atributos:
        class_names (dict): Mapeamento de labels para nomes das classes
        random_state (int): Semente para reprodutibilidade
    """
    
    def __init__(self, class_names: Dict[int, str] = None, random_state: int = 42):
        """
        Inicializa o avaliador.
        
        Args:
            class_names: Dicionário mapeando labels para nomes (ex: {0: 'Normal', 1: 'SQLi'})
            random_state: Semente para operações aleatórias
        """
        self.class_names = class_names or {0: 'Normal', 1: 'SQLi'}
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        
    def evaluate_model(
        self,
        model: Any,
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        X_train: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_train: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Executa avaliação completa do modelo.
        
        Args:
            model: Modelo treinado a ser avaliado
            X_test: Features de teste
            y_test: Labels de teste
            X_train: Features de treino (opcional para análise de learning curve)
            y_train: Labels de treino (opcional para análise de learning curve)
            
        Returns:
            Dicionário com resultados da avaliação contendo:
            - classification_metrics: Métricas de classificação
            - confusion_matrix: Matriz de confusão
            - roc_metrics: Métricas da curva ROC
            - pr_metrics: Métricas da curva Precision-Recall
            - feature_importance: Importância das features (se disponível)
            - learning_curve: Dados da learning curve (se X_train/y_train fornecidos)
        """
        results = {}
        
        try:
            # 1. Predições básicas
            y_pred = model.predict(X_test)
            y_proba = self._get_prediction_probabilities(model, X_test)
            
            # 2. Métricas de classificação
            results['classification_metrics'] = self._calculate_classification_metrics(y_test, y_pred)
            
            # 3. Matriz de confusão
            results['confusion_matrix'] = self._calculate_confusion_matrix(y_test, y_pred)
            
            # 4. Métricas ROC e Precision-Recall
            if y_proba is not None:
                results['roc_metrics'] = self._calculate_roc_metrics(y_test, y_proba)
                results['pr_metrics'] = self._calculate_pr_metrics(y_test, y_proba)
            
            # 5. Importância de features (se disponível)
            results['feature_importance'] = self._get_feature_importance(model, X_test)
            
            # 6. Learning curve (se dados de treino fornecidos)
            if X_train is not None and y_train is not None:
                results['learning_curve'] = self._calculate_learning_curve(model, X_train, y_train)
            
            self.logger.info("Avaliação do modelo concluída com sucesso")
            return results
            
        except Exception as e:
            self.logger.error(f"Erro durante avaliação do modelo: {str(e)}")
            raise

    def generate_report(
        self,
        evaluation_results: Dict[str, Any],
        output_dir: Union[str, Path] = None,
        model_name: str = "Modelo"
    ) -> Dict[str, str]:
        """
        Gera relatórios e visualizações da avaliação.
        
        Args:
            evaluation_results: Resultados da avaliação do modelo
            output_dir: Diretório para salvar os relatórios (opcional)
            model_name: Nome do modelo para os relatórios
            
        Returns:
            Dicionário com caminhos dos arquivos gerados
        """
        saved_files = {}
        output_dir = Path(output_dir) if output_dir else None
        
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. Salva métricas de classificação
            class_report = evaluation_results.get('classification_metrics', {})
            report_text = self._format_classification_report(class_report)
            
            if output_dir:
                report_path = output_dir / "classification_report.txt"
                with open(report_path, 'w') as f:
                    f.write(report_text)
                saved_files['classification_report'] = str(report_path)
            
            # 2. Gera e salva visualizações
            if output_dir:
                # Matriz de confusão
                cm_path = output_dir / "confusion_matrix.png"
                self._plot_confusion_matrix(
                    evaluation_results.get('confusion_matrix'),
                    save_path=cm_path,
                    model_name=model_name
                )
                saved_files['confusion_matrix'] = str(cm_path)
                
                # Curva ROC
                if 'roc_metrics' in evaluation_results:
                    roc_path = output_dir / "roc_curve.png"
                    self._plot_roc_curve(
                        evaluation_results['roc_metrics'],
                        save_path=roc_path,
                        model_name=model_name
                    )
                    saved_files['roc_curve'] = str(roc_path)
                
                # Curva Precision-Recall
                if 'pr_metrics' in evaluation_results:
                    pr_path = output_dir / "precision_recall_curve.png"
                    self._plot_pr_curve(
                        evaluation_results['pr_metrics'],
                        save_path=pr_path,
                        model_name=model_name
                    )
                    saved_files['precision_recall_curve'] = str(pr_path)
                
                # Importância de features
                if 'feature_importance' in evaluation_results:
                    fi_path = output_dir / "feature_importance.png"
                    self._plot_feature_importance(
                        evaluation_results['feature_importance'],
                        save_path=fi_path,
                        model_name=model_name
                    )
                    saved_files['feature_importance'] = str(fi_path)
                
                # Learning curve
                if 'learning_curve' in evaluation_results:
                    lc_path = output_dir / "learning_curve.png"
                    self._plot_learning_curve(
                        evaluation_results['learning_curve'],
                        save_path=lc_path,
                        model_name=model_name
                    )
                    saved_files['learning_curve'] = str(lc_path)
            
            return saved_files
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar relatório: {str(e)}")
            raise

    # Métodos auxiliares para cálculo de métricas
    def _get_prediction_probabilities(self, model, X_test) -> Optional[np.ndarray]:
        """Obtém probabilidades preditas, se suportado pelo modelo."""
        try:
            if hasattr(model, "predict_proba"):
                return model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                return model.decision_function(X_test)
            return None
        except Exception:
            return None

    def _calculate_classification_metrics(self, y_true, y_pred) -> Dict[str, Any]:
        """Calcula métricas de classificação."""
        report = classification_report(
            y_true, y_pred,
            target_names=list(self.class_names.values()),
            output_dict=True
        )
        
        # Adiciona acurácia balanceada para datasets desbalanceados
        report['balanced_accuracy'] = np.mean([
            report[str(label)]['recall'] 
            for label in self.class_names.keys()
        ])
        
        return report

    def _calculate_confusion_matrix(self, y_true, y_pred) -> np.ndarray:
        """Calcula a matriz de confusão."""
        return confusion_matrix(y_true, y_pred)

    def _calculate_roc_metrics(self, y_true, y_scores) -> Dict[str, Any]:
        """Calcula métricas da curva ROC."""
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        return {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': roc_auc,
            'optimal_idx': np.argmax(tpr - fpr),
            'optimal_threshold': thresholds[np.argmax(tpr - fpr)]
        }

    def _calculate_pr_metrics(self, y_true, y_scores) -> Dict[str, Any]:
        """Calcula métricas da curva Precision-Recall."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)
        
        return {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'average_precision': avg_precision,
            'optimal_idx': np.argmax(2 * precision * recall / (precision + recall + 1e-9)),
            'optimal_threshold': thresholds[np.argmax(2 * precision * recall / (precision + recall + 1e-9))]
        }

    def _get_feature_importance(self, model, feature_names) -> Optional[Dict[str, float]]:
        """Extrai importância das features, se disponível."""
        try:
            if hasattr(model, 'feature_importances_'):
                if isinstance(feature_names, pd.DataFrame):
                    feature_names = feature_names.columns.tolist()
                return dict(zip(feature_names, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                if isinstance(feature_names, pd.DataFrame):
                    feature_names = feature_names.columns.tolist()
                return dict(zip(feature_names, model.coef_[0]))
            return None
        except Exception:
            return None

    def _calculate_learning_curve(self, model, X_train, y_train) -> Dict[str, Any]:
        """Calcula dados para learning curve."""
        train_sizes, train_scores, test_scores = learning_curve(
            estimator=model,
            X=X_train,
            y=y_train,
            train_sizes=np.linspace(0.1, 1.0, 5),
            cv=3,
            scoring='f1',
            random_state=self.random_state
        )
        
        return {
            'train_sizes': train_sizes,
            'train_scores': train_scores,
            'test_scores': test_scores
        }

    # Métodos auxiliares para visualização
    def _plot_confusion_matrix(self, cm, save_path=None, model_name=""):
        """Plota matriz de confusão com estilo profissional."""
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Usando ConfusionMatrixDisplay do scikit-learn
            display = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=list(self.class_names.values())
            )
            
            # Configurações do plot
            display.plot(
                cmap='Blues', 
                ax=ax,
                values_format='d',  # Formato inteiro
                colorbar=False
            )
            
            # Ajustes estéticos
            title = f'Matriz de Confusão - {model_name}' if model_name else 'Matriz de Confusão'
            ax.set_title(title, pad=20)
            ax.grid(False)
            
            # Melhora a legibilidade dos textos
            for text in ax.texts:
                text.set_size(12)
                if cm.sum() > 0:  # Evita divisão por zero
                    text.set_color('white' if int(text.get_text())/cm.sum() > 0.5 else 'black')
            
            # Salva ou mostra a figura
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            self.logger.error(f"Erro ao plotar matriz de confusão: {str(e)}")
            raise

    def _plot_roc_curve(self, roc_metrics, save_path=None, model_name=""):
        """Plota curva ROC."""
        fig, ax = plt.subplots(figsize=(8, 6))
        RocCurveDisplay(
            fpr=roc_metrics['fpr'],
            tpr=roc_metrics['tpr'],
            roc_auc=roc_metrics['auc']
        ).plot(ax=ax)
        ax.set_title(f'Curva ROC - {model_name}')
        ax.plot([0, 1], [0, 1], 'k--')  # Linha de referência
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

    def _plot_pr_curve(self, pr_metrics, save_path=None, model_name=""):
        """Plota curva Precision-Recall."""
        fig, ax = plt.subplots(figsize=(8, 6))
        PrecisionRecallDisplay(
            precision=pr_metrics['precision'],
            recall=pr_metrics['recall'],
            average_precision=pr_metrics['average_precision']
        ).plot(ax=ax)
        ax.set_title(f'Curva Precision-Recall - {model_name}')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

    def _plot_feature_importance(self, feature_importance, save_path=None, model_name="", top_n=20):
        """Plota importância das features."""
        if not feature_importance:
            return
            
        # Ordena as features por importância
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        names, values = zip(*sorted_features[:top_n])
        
        fig, ax = plt.subplots(figsize=(10, 8))
        y_pos = np.arange(len(names))
        ax.barh(y_pos, values, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel('Importância')
        ax.set_title(f'Top {top_n} Features - {model_name}')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

    def _plot_learning_curve(self, lc_data, save_path=None, model_name=""):
        """Plota learning curve."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        train_sizes = lc_data['train_sizes']
        train_scores_mean = np.mean(lc_data['train_scores'], axis=1)
        train_scores_std = np.std(lc_data['train_scores'], axis=1)
        test_scores_mean = np.mean(lc_data['test_scores'], axis=1)
        test_scores_std = np.std(lc_data['test_scores'], axis=1)
        
        ax.fill_between(
            train_sizes,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.1,
            color="r"
        )
        ax.fill_between(
            train_sizes,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color="g"
        )
        ax.plot(
            train_sizes,
            train_scores_mean,
            'o-',
            color="r",
            label="Training score"
        )
        ax.plot(
            train_sizes,
            test_scores_mean,
            'o-',
            color="g",
            label="Cross-validation score"
        )
        
        ax.set_xlabel("Training examples")
        ax.set_ylabel("F1 Score")
        ax.legend(loc="best")
        ax.set_title(f"Learning Curve - {model_name}")
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

    def _format_classification_report(self, report) -> str:
        """Formata o relatório de classificação para saída legível."""
        lines = ["=== Classification Report ==="]
        
        # Adiciona métricas por classe
        for class_name in self.class_names.values():
            if class_name in report:
                lines.append(f"\nClass {class_name}:")
                lines.append(f"  Precision: {report[class_name]['precision']:.4f}")
                lines.append(f"  Recall:    {report[class_name]['recall']:.4f}")
                lines.append(f"  F1-score:  {report[class_name]['f1-score']:.4f}")
                lines.append(f"  Support:   {report[class_name]['support']}")
        
        # Adiciona métricas agregadas
        lines.append("\nAggregated Metrics:")
        lines.append(f"  Accuracy:           {report['accuracy']:.4f}")
        lines.append(f"  Balanced Accuracy:  {report.get('balanced_accuracy', 0):.4f}")
        lines.append(f"  Macro Avg F1-score: {report['macro avg']['f1-score']:.4f}")
        lines.append(f"  Weighted Avg F1:    {report['weighted avg']['f1-score']:.4f}")
        
        return "\n".join(lines)