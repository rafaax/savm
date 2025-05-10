import logging
from typing import Optional, Dict, Any, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

class ModelVisualizer:
    """
    Classe especializada em visualizações para avaliação de modelos de detecção de SQL injection.
    
    Configura estilos profissionais e gera visualizações para:
    - Matriz de confusão
    - Curvas ROC e Precision-Recall
    - Importância de features
    - Learning curves
    - Calibration plots
    - Distribuição de probabilidades
    """

    def __init__(self):
        self._set_visualization_styles()
        self.logger = logging.getLogger(__name__)

    def _set_visualization_styles(self):
        """Configura estilos visuais consistentes para todos os gráficos."""
        plt.style.use('seaborn')
        sns.set_context("notebook")
        self.PALETTE = sns.color_palette("husl", 8)
        
        self.STYLE_CONFIG = {
            'figure.figsize': (10, 7),
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'font.family': 'DejaVu Sans'
        }
        
        plt.rcParams.update(self.STYLE_CONFIG)

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: Dict[int, str],
        save_path: Optional[Union[str, Path]] = None,
        model_name: str = "",
        normalize: bool = True,
        dpi: int = 300
    ) -> plt.Figure:
        """
        Plota matriz de confusão com anotações e estilo profissional.
        
        Args:
            cm: Matriz de confusão numpy array
            class_names: Mapeamento de labels para nomes de classes
            save_path: Caminho para salvar a figura (opcional)
            model_name: Nome do modelo para o título
            normalize: Se True, normaliza por linha
            dpi: Resolução da imagem
            
        Returns:
            Objeto Figure do matplotlib
        """
        try:
            # Normalização
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                fmt = '.2f'
            else:
                fmt = 'd'

            # Cria figura
            fig, ax = plt.subplots(figsize=self.STYLE_CONFIG['figure.figsize'])
            
            # Configura visualização
            display = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=list(class_names.values())
            )
            
            # Plota com estilo
            display.plot(
                cmap='Blues',
                ax=ax,
                values_format=fmt,
                colorbar=False
            )
            
            # Adiciona título e ajustes
            title = f"Matriz de Confusão{' Normalizada' if normalize else ''}"
            if model_name:
                title += f" - {model_name}"
                
            ax.set_title(title, pad=20)
            ax.grid(False)
            
            # Melhora legibilidade
            for text in ax.texts:
                text.set_size(12)
                text.set_color('black' if float(text.get_text()) < 0.7 else 'white')

            # Salva ou mostra
            if save_path:
                self._save_figure(fig, save_path, dpi)
            else:
                plt.show()
                
            return fig
            
        except Exception as e:
            self.logger.error(f"Erro ao plotar matriz de confusão: {str(e)}")
            raise

    def plot_roc_curve(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        roc_auc: float,
        save_path: Optional[Union[str, Path]] = None,
        model_name: str = "",
        dpi: int = 300
    ) -> plt.Figure:
        """
        Plota curva ROC com estilo profissional.
        
        Args:
            fpr: Taxa de falsos positivos
            tpr: Taxa de verdadeiros positivos
            roc_auc: Valor AUC-ROC
            save_path: Caminho para salvar a figura (opcional)
            model_name: Nome do modelo para o título
            dpi: Resolução da imagem
            
        Returns:
            Objeto Figure do matplotlib
        """
        try:
            fig, ax = plt.subplots(figsize=self.STYLE_CONFIG['figure.figsize'])
            
            # Plota curva ROC
            display = RocCurveDisplay(
                fpr=fpr,
                tpr=tpr,
                roc_auc=roc_auc
            )
            display.plot(ax=ax, color=self.PALETTE[0])
            
            # Linha de referência
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            
            # Configurações estéticas
            title = f"Curva ROC (AUC = {roc_auc:.3f})"
            if model_name:
                title += f" - {model_name}"
                
            ax.set_title(title, pad=20)
            ax.set_xlim([-0.01, 1.01])
            ax.set_ylim([-0.01, 1.01])
            ax.grid(True, alpha=0.3)
            
            # Salva ou mostra
            if save_path:
                self._save_figure(fig, save_path, dpi)
            else:
                plt.show()
                
            return fig
            
        except Exception as e:
            self.logger.error(f"Erro ao plotar curva ROC: {str(e)}")
            raise

    def plot_precision_recall_curve(
        self,
        precision: np.ndarray,
        recall: np.ndarray,
        average_precision: float,
        save_path: Optional[Union[str, Path]] = None,
        model_name: str = "",
        dpi: int = 300
    ) -> plt.Figure:
        """
        Plota curva Precision-Recall com estilo profissional.
        
        Args:
            precision: Valores de precisão
            recall: Valores de recall
            average_precision: Valor AP
            save_path: Caminho para salvar a figura (opcional)
            model_name: Nome do modelo para o título
            dpi: Resolução da imagem
            
        Returns:
            Objeto Figure do matplotlib
        """
        try:
            fig, ax = plt.subplots(figsize=self.STYLE_CONFIG['figure.figsize'])
            
            # Plota curva Precision-Recall
            display = PrecisionRecallDisplay(
                precision=precision,
                recall=recall,
                average_precision=average_precision
            )
            display.plot(ax=ax, color=self.PALETTE[1])
            
            # Configurações estéticas
            title = f"Curva Precision-Recall (AP = {average_precision:.3f})"
            if model_name:
                title += f" - {model_name}"
                
            ax.set_title(title, pad=20)
            ax.set_xlim([-0.01, 1.01])
            ax.set_ylim([-0.01, 1.01])
            ax.grid(True, alpha=0.3)
            
            # Salva ou mostra
            if save_path:
                self._save_figure(fig, save_path, dpi)
            else:
                plt.show()
                
            return fig
            
        except Exception as e:
            self.logger.error(f"Erro ao plotar curva Precision-Recall: {str(e)}")
            raise

    def plot_feature_importance(
        self,
        feature_importance: Dict[str, float],
        save_path: Optional[Union[str, Path]] = None,
        model_name: str = "",
        top_n: int = 20,
        dpi: int = 300
    ) -> plt.Figure:
        """
        Plota importância de features com estilo profissional.
        
        Args:
            feature_importance: Dicionário com nomes e valores de importância
            save_path: Caminho para salvar a figura (opcional)
            model_name: Nome do modelo para o título
            top_n: Número de features principais a mostrar
            dpi: Resolução da imagem
            
        Returns:
            Objeto Figure do matplotlib
        """
        try:
            if not feature_importance:
                self.logger.warning("Nenhuma importância de feature disponível")
                return None
                
            # Prepara dados
            features = sorted(feature_importance.items(), 
                            key=lambda x: abs(x[1]), 
                            reverse=True)[:top_n]
            names, values = zip(*features)
            
            # Cria figura
            fig, ax = plt.subplots(figsize=(10, 0.5 * len(names)))
            
            # Plota barras horizontais
            y_pos = np.arange(len(names))
            bars = ax.barh(y_pos, values, color=self.PALETTE[2], alpha=0.7)
            
            # Configurações estéticas
            title = f"Top {top_n} Features por Importância"
            if model_name:
                title += f" - {model_name}"
                
            ax.set_title(title, pad=20)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names)
            ax.invert_yaxis()
            ax.set_xlabel("Importância")
            ax.grid(True, axis='x', alpha=0.3)
            
            # Adiciona valores nas barras
            for bar in bars:
                width = bar.get_width()
                ax.text(width * 1.02, bar.get_y() + bar.get_height()/2.,
                        f'{width:.3f}',
                        va='center', ha='left')
            
            # Salva ou mostra
            if save_path:
                self._save_figure(fig, save_path, dpi)
            else:
                plt.show()
                
            return fig
            
        except Exception as e:
            self.logger.error(f"Erro ao plotar importância de features: {str(e)}")
            raise

    def plot_learning_curve(
        self,
        train_sizes: np.ndarray,
        train_scores: np.ndarray,
        test_scores: np.ndarray,
        save_path: Optional[Union[str, Path]] = None,
        model_name: str = "",
        metric_name: str = "F1 Score",
        dpi: int = 300
    ) -> plt.Figure:
        """
        Plota learning curve com estilo profissional.
        
        Args:
            train_sizes: Tamanhos dos conjuntos de treino
            train_scores: Scores no conjunto de treino
            test_scores: Scores no conjunto de validação
            save_path: Caminho para salvar a figura (opcional)
            model_name: Nome do modelo para o título
            metric_name: Nome da métrica sendo plotada
            dpi: Resolução da imagem
            
        Returns:
            Objeto Figure do matplotlib
        """
        try:
            fig, ax = plt.subplots(figsize=self.STYLE_CONFIG['figure.figsize'])
            
            # Calcula médias e desvios padrão
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            
            # Plota áreas de desvio padrão
            ax.fill_between(
                train_sizes,
                train_mean - train_std,
                train_mean + train_std,
                alpha=0.1,
                color=self.PALETTE[3]
            )
            ax.fill_between(
                train_sizes,
                test_mean - test_std,
                test_mean + test_std,
                alpha=0.1,
                color=self.PALETTE[4]
            )
            
            # Plota linhas das médias
            ax.plot(
                train_sizes,
                train_mean,
                'o-',
                color=self.PALETTE[3],
                label="Treino"
            )
            ax.plot(
                train_sizes,
                test_mean,
                'o-',
                color=self.PALETTE[4],
                label="Validação"
            )
            
            # Configurações estéticas
            title = f"Learning Curve - {metric_name}"
            if model_name:
                title += f" - {model_name}"
                
            ax.set_title(title, pad=20)
            ax.set_xlabel("Tamanho do Conjunto de Treino")
            ax.set_ylabel(metric_name)
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            
            # Salva ou mostra
            if save_path:
                self._save_figure(fig, save_path, dpi)
            else:
                plt.show()
                
            return fig
            
        except Exception as e:
            self.logger.error(f"Erro ao plotar learning curve: {str(e)}")
            raise

    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        save_path: Optional[Union[str, Path]] = None,
        model_name: str = "",
        n_bins: int = 10,
        dpi: int = 300
    ) -> plt.Figure:
        """
        Plota curva de calibração para avaliar probabilidades preditas.
        
        Args:
            y_true: Labels verdadeiros
            y_prob: Probabilidades preditas
            save_path: Caminho para salvar a figura (opcional)
            model_name: Nome do modelo para o título
            n_bins: Número de bins para discretização
            dpi: Resolução da imagem
            
        Returns:
            Objeto Figure do matplotlib
        """
        try:
            fig, ax = plt.subplots(figsize=self.STYLE_CONFIG['figure.figsize'])
            
            # Calcula curva de calibração
            prob_true, prob_pred = calibration_curve(
                y_true, y_prob, n_bins=n_bins
            )
            
            # Plota curva de calibração
            ax.plot(prob_pred, prob_true, 's-', color=self.PALETTE[5], 
                   label=f"Modelo (Bins={n_bins})")
            
            # Linha de referência (calibração perfeita)
            ax.plot([0, 1], [0, 1], 'k:', label="Calibração Perfeita")
            
            # Configurações estéticas
            title = "Curva de Calibração"
            if model_name:
                title += f" - {model_name}"
                
            ax.set_title(title, pad=20)
            ax.set_xlabel("Probabilidade Média Predita")
            ax.set_ylabel("Fração de Positivos")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            
            # Salva ou mostra
            if save_path:
                self._save_figure(fig, save_path, dpi)
            else:
                plt.show()
                
            return fig
            
        except Exception as e:
            self.logger.error(f"Erro ao plotar curva de calibração: {str(e)}")
            raise

    def plot_probability_distribution(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        class_names: Dict[int, str],
        save_path: Optional[Union[str, Path]] = None,
        model_name: str = "",
        dpi: int = 300
    ) -> plt.Figure:
        """
        Plota distribuição de probabilidades preditas por classe real.
        
        Args:
            y_true: Labels verdadeiros
            y_prob: Probabilidades preditas
            class_names: Mapeamento de labels para nomes de classes
            save_path: Caminho para salvar a figura (opcional)
            model_name: Nome do modelo para o título
            dpi: Resolução da imagem
            
        Returns:
            Objeto Figure do matplotlib
        """
        try:
            fig, ax = plt.subplots(figsize=self.STYLE_CONFIG['figure.figsize'])
            
            # Cria DataFrame para facilitar o plotting
            plot_data = pd.DataFrame({
                'Probabilidade Predita': y_prob,
                'Classe Real': [class_names[y] for y in y_true]
            })
            
            # Plota distribuição com KDE
            sns.histplot(
                data=plot_data,
                x='Probabilidade Predita',
                hue='Classe Real',
                element='step',
                stat='density',
                common_norm=False,
                kde=True,
                alpha=0.3,
                palette=[self.PALETTE[6], self.PALETTE[7]]
            )
            
            # Configurações estéticas
            title = "Distribuição de Probabilidades Preditas"
            if model_name:
                title += f" - {model_name}"
                
            ax.set_title(title, pad=20)
            ax.set_xlim(0, 1)
            ax.grid(True, alpha=0.3)
            
            # Salva ou mostra
            if save_path:
                self._save_figure(fig, save_path, dpi)
            else:
                plt.show()
                
            return fig
            
        except Exception as e:
            self.logger.error(f"Erro ao plotar distribuição de probabilidades: {str(e)}")
            raise

    def _save_figure(self, fig: plt.Figure, save_path: Union[str, Path], dpi: int = 300):
        """
        Salva figura com tratamento de erros e configurações consistentes.
        
        Args:
            fig: Figura matplotlib
            save_path: Caminho para salvar
            dpi: Resolução da imagem
        """
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, bbox_inches='tight', dpi=dpi)
            plt.close(fig)
            self.logger.info(f"Figura salva em: {save_path}")
        except Exception as e:
            self.logger.error(f"Erro ao salvar figura: {str(e)}")
            raise