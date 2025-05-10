import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from pathlib import Path
import re

class DataValidator:
    """
    Classe para validação avançada de datasets de detecção de SQL injection.
    
    Atributos:
        min_query_length (int): Comprimento mínimo aceitável para queries
        max_query_length (int): Comprimento máximo aceitável para queries
        allowed_special_chars (set): Caracteres especiais permitidos
    """
    
    def __init__(self):
        self.min_query_length = 1
        self.max_query_length = 10000
        self.allowed_special_chars = {"'", '"', ";", "--", "/*", "*/", "#", "@", "*", "="}
        self.logger = logging.getLogger(__name__)
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Executa validação completa no dataset.
        
        Args:
            df: DataFrame contendo os dados a serem validados
            
        Returns:
            Dicionário com resultados da validação contendo:
            - is_valid: bool indicando se os dados são válidos
            - stats: estatísticas do dataset
            - issues: problemas encontrados
        """
        validation_result = {
            'is_valid': True,
            'stats': {},
            'issues': {}
        }
        
        try:
            # Validação estrutural
            structure_issues = self._validate_structure(df)
            if structure_issues:
                validation_result['issues'].update({'structure': structure_issues})
                validation_result['is_valid'] = False
            
            # Análise de conteúdo
            content_stats, content_issues = self._validate_content(df)
            validation_result['stats'].update(content_stats)
            if content_issues:
                validation_result['issues'].update({'content': content_issues})
                validation_result['is_valid'] = False
            
            # Validação de distribuição
            distribution_issues = self._validate_distribution(df)
            if distribution_issues:
                validation_result['issues'].update({'distribution': distribution_issues})
            
            self.logger.info("Validação do dataset concluída")
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Erro durante validação: {str(e)}")
            validation_result['is_valid'] = False
            validation_result['issues']['exception'] = str(e)
            return validation_result
    
    def _validate_structure(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Valida a estrutura básica do DataFrame"""
        issues = {}
        
        # Verifica colunas obrigatórias
        required_columns = {'query', 'label'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            issues['missing_columns'] = list(missing_columns)
        
        # Verifica tipos de dados
        type_issues = []
        if 'query' in df.columns and not pd.api.types.is_string_dtype(df['query']):
            type_issues.append("Coluna 'query' deve ser do tipo string")
        if 'label' in df.columns and not pd.api.types.is_numeric_dtype(df['label']):
            type_issues.append("Coluna 'label' deve ser numérica")
            
        if type_issues:
            issues['type_issues'] = type_issues
            
        return issues
    
    def _validate_content(self, df: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
        """Valida o conteúdo das queries e labels"""
        stats = {}
        issues = {}
        
        # Comprimento das queries
        query_lengths = df['query'].str.len()
        stats['query_length_stats'] = {
            'min': query_lengths.min(),
            'max': query_lengths.max(),
            'mean': query_lengths.mean(),
            'median': query_lengths.median()
        }
        
        # Queries muito curtas/longas
        short_queries = df[query_lengths < self.min_query_length]
        long_queries = df[query_lengths > self.max_query_length]
        
        if not short_queries.empty:
            issues['short_queries'] = list(short_queries.index[:5])  # Mostra primeiros 5 exemplos
        if not long_queries.empty:
            issues['long_queries'] = list(long_queries.index[:5])
        
        # Labels inválidas
        valid_labels = {0, 1}
        invalid_labels = df[~df['label'].isin(valid_labels)]
        if not invalid_labels.empty:
            issues['invalid_labels'] = {
                'count': len(invalid_labels),
                'examples': invalid_labels['label'].unique().tolist()
            }
        
        return stats, issues
    
    def _validate_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Valida a distribuição das classes"""
        issues = {}
        
        class_distribution = df['label'].value_counts(normalize=True)
        minority_class = class_distribution.idxmin()
        minority_ratio = class_distribution.min()
        
        if minority_ratio < 0.2:  # Alerta se classe minoritária < 20%
            issues['class_imbalance'] = {
                'minority_class': minority_class,
                'minority_ratio': minority_ratio,
                'suggestion': 'Considerar técnicas de balanceamento'
            }
        
        return issues
    
    def detect_special_characters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detecta caracteres especiais nas queries.
        
        Args:
            df: DataFrame contendo coluna 'query'
            
        Returns:
            DataFrame com colunas adicionais indicando presença de caracteres especiais
        """
        special_char_cols = {}
        
        for char in self.allowed_special_chars:
            col_name = f'has_{char.replace(" ", "_").replace("/", "_")}'
            special_char_cols[col_name] = df['query'].str.contains(re.escape(char), regex=True)
        
        return df.assign(**special_char_cols)
    
    def generate_validation_report(self, validation_result: Dict[str, Any], 
                                 report_path: Optional[str] = None) -> str:
        """
        Gera um relatório de validação em formato legível.
        
        Args:
            validation_result: Resultado da validação
            report_path: Caminho para salvar o relatório (opcional)
            
        Returns:
            String com o relatório formatado
        """
        report_lines = [
            "=== Relatório de Validação de Dados ===",
            f"Status Geral: {'VÁLIDO' if validation_result['is_valid'] else 'INVÁLIDO'}",
            "\n--- Estatísticas ---"
        ]
        
        # Adiciona estatísticas
        for stat_name, stat_value in validation_result['stats'].items():
            report_lines.append(f"\n* {stat_name}:")
            if isinstance(stat_value, dict):
                for k, v in stat_value.items():
                    report_lines.append(f"  - {k}: {v}")
            else:
                report_lines.append(f"  - {stat_value}")
        
        # Adiciona problemas encontrados
        if validation_result['issues']:
            report_lines.append("\n--- Problemas Encontrados ---")
            for issue_type, details in validation_result['issues'].items():
                report_lines.append(f"\n* {issue_type.replace('_', ' ').title()}:")
                if isinstance(details, dict):
                    for k, v in details.items():
                        report_lines.append(f"  - {k}: {v}")
                elif isinstance(details, list):
                    report_lines.append(f"  - Quantidade: {len(details)}")
                    report_lines.append(f"  - Exemplos: {details[:5]}{'...' if len(details)>5 else ''}")
                else:
                    report_lines.append(f"  - {details}")
        
        report_text = "\n".join(report_lines)
        
        if report_path:
            try:
                with open(report_path, 'w') as f:
                    f.write(report_text)
                self.logger.info(f"Relatório salvo em {report_path}")
            except Exception as e:
                self.logger.error(f"Erro ao salvar relatório: {str(e)}")
        
        return report_text