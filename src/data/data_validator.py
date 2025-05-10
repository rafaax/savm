import pandas as pd
import logging
import re
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

class DataValidator:
    """
    Classe para validação avançada de datasets de detecção de SQL injection.
    
    Atributos:
        min_query_length (int): Comprimento mínimo aceitável para queries
        max_query_length (int): Comprimento máximo aceitável para queries
        allowed_special_chars (set): Caracteres especiais permitidos
    """
    
    def __init__(self, min_query_length: int = 1, max_query_length: int = 10000):
        """
        Inicializa o validador com configurações personalizáveis.
        
        Args:
            min_query_length: Comprimento mínimo aceitável para queries
            max_query_length: Comprimento máximo aceitável para queries
        """
        self.min_query_length = min_query_length
        self.max_query_length = max_query_length
        self.allowed_special_chars = {"'", '"', ";", "--", "/*", "*/", "#", "@", "*", "=", "(", ")", "<", ">"}
        self.logger = logging.getLogger(__name__)
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Executa validação completa no dataset.
        
        Args:
            df: DataFrame contendo os dados a serem validados
            
        Returns:
            Dicionário com:
            - is_valid (bool): Se os dados são válidos
            - stats (dict): Estatísticas do dataset
            - issues (dict): Problemas encontrados
            - warnings (dict): Alertas não críticos
        """
        validation_result = {
            'is_valid': True,
            'stats': {},
            'issues': {},
            'warnings': {}
        }
        
        try:
            # Validação estrutural (crítica)
            structure_issues = self._validate_structure(df)
            if structure_issues:
                validation_result['issues']['structure'] = structure_issues
                validation_result['is_valid'] = False
            
            # Análise de conteúdo (crítica)
            content_stats, content_issues = self._validate_content(df)
            validation_result['stats'].update(content_stats)
            if content_issues:
                validation_result['issues']['content'] = content_issues
                validation_result['is_valid'] = False
            
            # Validação de distribuição (apenas alerta)
            distribution_issues = self._validate_distribution(df)
            if distribution_issues:
                validation_result['warnings']['distribution'] = distribution_issues
            
            # Verificação de caracteres especiais (informativo)
            special_chars_stats = self._check_special_characters(df)
            validation_result['stats']['special_chars'] = special_chars_stats
            
            self.logger.info("Validação concluída com %s",
                           "sucesso" if validation_result['is_valid'] else "erros")
            return validation_result
            
        except Exception as e:
            self.logger.error("Erro durante validação: %s", str(e), exc_info=True)
            validation_result.update({
                'is_valid': False,
                'issues': {'exception': str(e)}
            })
            return validation_result
    
    def _validate_structure(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Valida a estrutura básica do DataFrame."""
        issues = {}
        
        # Verifica colunas obrigatórias
        required_columns = {'query', 'label'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            issues['missing_columns'] = sorted(missing_columns)
        
        # Verifica tipos de dados
        type_issues = []
        if 'query' in df.columns and not pd.api.types.is_string_dtype(df['query']):
            type_issues.append("Coluna 'query' deve ser do tipo string")
        if 'label' in df.columns and not pd.api.types.is_numeric_dtype(df['label']):
            type_issues.append("Coluna 'label' deve ser numérica")
        
        if type_issues:
            issues['type_issues'] = type_issues
            
        return issues
    
    def _validate_content(self, df: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Valida o conteúdo das queries e labels."""
        stats = {}
        issues = {}
        
        # Análise de comprimento das queries
        df = df.copy()
        df['query_length'] = df['query'].str.len()
        
        query_lengths = df['query_length']
        stats['query_length'] = {
            'min': int(query_lengths.min()),
            'max': int(query_lengths.max()),
            'mean': float(query_lengths.mean()),
            'median': float(query_lengths.median()),
            'std': float(query_lengths.std())
        }
        
        # Identifica queries inválidas
        invalid_queries = {}
        short_queries = df[query_lengths < self.min_query_length]
        long_queries = df[query_lengths > self.max_query_length]
        
        if not short_queries.empty:
            invalid_queries['short'] = {
                'count': len(short_queries),
                'examples': short_queries['query'].head(3).tolist()
            }
        if not long_queries.empty:
            invalid_queries['long'] = {
                'count': len(long_queries),
                'examples': long_queries['query'].head(3).tolist()
            }
        
        if invalid_queries:
            issues['invalid_queries'] = invalid_queries
        
        # Valida labels
        valid_labels = {0, 1}
        invalid_labels = df[~df['label'].isin(valid_labels)]
        
        if not invalid_labels.empty:
            issues['invalid_labels'] = {
                'count': len(invalid_labels),
                'invalid_values': invalid_labels['label'].unique().tolist(),
                'examples': invalid_labels.head(3).to_dict('records')
            }
        
        return stats, issues
    
    def _validate_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Verifica o balanceamento das classes."""
        class_dist = df['label'].value_counts(normalize=True).to_dict()
        minority_class = min(class_dist, key=class_dist.get)
        minority_ratio = class_dist[minority_class]
        
        if minority_ratio < 0.2:  # Alerta se classe minoritária < 20%
            return {
                'class_imbalance': {
                    'minority_class': minority_class,
                    'minority_ratio': round(minority_ratio, 4),
                    'suggestion': 'Considerar oversampling/undersampling'
                }
            }
        return {}
    
    def _check_special_characters(self, df: pd.DataFrame) -> Dict[str, int]:
        """Analisa a frequência de caracteres especiais."""
        char_stats = {}
        query_sample = df['query'].str.cat(sep=' ')
        
        for char in self.allowed_special_chars:
            count = query_sample.count(char)
            if count > 0:
                char_stats[char] = count
        
        return char_stats
    
    def generate_validation_report(self, 
                                validation_result: Dict[str, Any], 
                                report_path: Optional[Union[str, Path]] = None) -> str:
        """
        Gera um relatório detalhado de validação.
        
        Args:
            validation_result: Resultado da validação
            report_path: Caminho opcional para salvar o relatório
            
        Returns:
            Relatório formatado como string
        """
        report = [
            "=== Relatório de Validação de Dados ===",
            f"Status Geral: {'VÁLIDO' if validation_result['is_valid'] else 'INVÁLIDO'}"
        ]
        
        # Seção de estatísticas
        report.append("\n=== Estatísticas ===")
        for category, stats in validation_result['stats'].items():
            report.append(f"\n{category.upper()}:")
            if isinstance(stats, dict):
                for k, v in stats.items():
                    report.append(f"  - {k}: {v}")
            else:
                report.append(f"  - {stats}")
        
        # Seção de problemas
        if validation_result['issues']:
            report.append("\n=== PROBLEMAS ENCONTRADOS ===")
            for issue_type, details in validation_result['issues'].items():
                report.append(f"\n* {issue_type.replace('_', ' ').title()}:")
                self._format_report_details(report, details)
        
        # Seção de alertas
        if validation_result['warnings']:
            report.append("\n=== ALERTAS ===")
            for warning_type, details in validation_result['warnings'].items():
                report.append(f"\n* {warning_type.replace('_', ' ').title()}:")
                self._format_report_details(report, details)
        
        full_report = "\n".join(report)
        
        # Salva em arquivo se especificado
        if report_path:
            try:
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(full_report)
                self.logger.info("Relatório salvo em %s", report_path)
            except Exception as e:
                self.logger.error("Erro ao salvar relatório: %s", str(e))
        
        return full_report
    
    def _format_report_details(self, report: List[str], details: Any, indent: str = "  ") -> None:
        """Auxilia na formatação de detalhes do relatório."""
        if isinstance(details, dict):
            for k, v in details.items():
                if isinstance(v, (list, dict)) and v:
                    report.append(f"{indent}- {k}:")
                    self._format_report_details(report, v, indent + "  ")
                else:
                    report.append(f"{indent}- {k}: {v}")
        elif isinstance(details, list):
            for item in details[:5]:  # Limita a 5 exemplos
                report.append(f"{indent}- {item}")
            if len(details) > 5:
                report.append(f"{indent}- ... ({len(details)-5} itens omitidos)")
        else:
            report.append(f"{indent}- {details}")