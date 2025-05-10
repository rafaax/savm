import pandas as pd
import logging
from pathlib import Path
from typing import Union, Optional, Dict, Any
import json

class DataLoader:
    """
    Classe para carregar e validar dados para detecção de SQL injection.
    
    Atributos:
        expected_columns (list): Lista de colunas obrigatórias
        valid_labels (list): Valores aceitáveis para a coluna de label
    """
    
    def __init__(self):
        self.expected_columns = ['query', 'label']
        self.valid_labels = [0, 1]  # 0 = legítimo, 1 = malicioso
        self.logger = logging.getLogger(__name__)
        
    def load_csv(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Carrega dados de um arquivo CSV com validação.
        
        Args:
            file_path: Caminho para o arquivo CSV
            **kwargs: Argumentos adicionais para pd.read_csv()
            
        Returns:
            DataFrame com os dados carregados e validados
            
        Raises:
            ValueError: Se o arquivo ou dados forem inválidos
            FileNotFoundError: Se o arquivo não existir
        """
        try:
            self.logger.info(f"Carregando dados de {file_path}")
            
            # Verifica se o arquivo existe
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
            
            # Carrega os dados
            df = pd.read_csv(file_path, **kwargs)
            
            # Validação básica
            self._validate_data(df)
            
            # Pré-processamento inicial
            df = self._basic_preprocessing(df)
            
            self.logger.info(f"Dados carregados com sucesso. Total de registros: {len(df)}")
            return df
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar dados: {str(e)}")
            raise

    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Valida a estrutura e conteúdo dos dados.
        
        Args:
            df: DataFrame a ser validado
            
        Raises:
            ValueError: Se os dados não atenderem aos requisitos
        """
        # Verifica colunas obrigatórias
        missing_cols = [col for col in self.expected_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Colunas obrigatórias ausentes: {missing_cols}")
        
        # Verifica valores da label
        invalid_labels = [label for label in df['label'].unique() 
                         if label not in self.valid_labels]
        if invalid_labels:
            raise ValueError(f"Valores inválidos na coluna 'label': {invalid_labels}")
        
        # Verifica queries vazias
        empty_queries = df['query'].isna().sum()
        if empty_queries > 0:
            self.logger.warning(f"Encontradas {empty_queries} queries vazias. Serão removidas.")
    
    def _basic_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica pré-processamento básico nos dados.
        
        Args:
            df: DataFrame com os dados brutos
            
        Returns:
            DataFrame pré-processado
        """
        # Remove linhas com queries vazias
        df = df.dropna(subset=['query'])
        
        # Remove espaços extras
        df['query'] = df['query'].str.strip()
        
        # Converte labels para inteiro (caso estejam como string)
        df['label'] = df['label'].astype(int)
        
        return df

    def save_processed_data(self, df: pd.DataFrame, output_path: Union[str, Path], 
                           format: str = 'csv') -> None:
        """
        Salva dados processados em diferentes formatos.
        
        Args:
            df: DataFrame a ser salvo
            output_path: Caminho de saída
            format: Formato de saída (csv, parquet, json)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format == 'csv':
                df.to_csv(output_path, index=False)
            elif format == 'parquet':
                df.to_parquet(output_path, index=False)
            elif format == 'json':
                df.to_json(output_path, orient='records', lines=True)
            else:
                raise ValueError(f"Formato não suportado: {format}")
                
            self.logger.info(f"Dados salvos em {output_path} como {format}")
        except Exception as e:
            self.logger.error(f"Erro ao salvar dados: {str(e)}")
            raise

    def load_from_json(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Carrega dados de um arquivo JSON.
        
        Args:
            file_path: Caminho para o arquivo JSON
            
        Returns:
            DataFrame com os dados carregados
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return pd.DataFrame(data)
        except Exception as e:
            self.logger.error(f"Erro ao carregar JSON: {str(e)}")
            raise