import os
from src.sqli.sqli_detector import SQLIDetector


def loadModelSqli(caminho_modelo_para_carregar: str | None) -> 'SQLIDetector | None':
    """
    Tenta carregar um modelo SQLIDetector de um caminho especificado e valida se está treinado.

    Args:
        caminho_modelo_para_carregar (str | None): O caminho completo para o arquivo do modelo.

    Returns:
        SQLIDetector | None: A instância do modelo carregado e treinado, ou None se falhar.
    """
    if not caminho_modelo_para_carregar:
        # Não há caminho para carregar (pode ser porque o arquivo de info não foi encontrado ou estava vazio)
        return None

    print(f"API: Tentando carregar o modelo SQLi de: {caminho_modelo_para_carregar}")
    
    if not os.path.exists(caminho_modelo_para_carregar):
        print(f"API ERRO: Arquivo do modelo '{caminho_modelo_para_carregar}' (indicado como o mais recente) não encontrado.")
        return None

    try:
        detector_instance = SQLIDetector.load_model(caminho_modelo_para_carregar) # Supondo que SQLIDetector.load_model existe
        
        if not detector_instance:
            # load_model pode retornar None se falhar internamente (ex: arquivo corrompido, tipo errado)
            print(f"API ERRO: Falha ao carregar o modelo de '{caminho_modelo_para_carregar}'. Verifique os logs do SQLIDetector.load_model.")
            return None

        if detector_instance.is_trained(): # Supondo que detector_instance.is_trained() existe
            print("API: Modelo SQLi pré-treinado carregado com sucesso.")
            return detector_instance
        else:
            print(f"API ERRO: Modelo carregado de '{caminho_modelo_para_carregar}', mas não está marcado como treinado! "
                  "Por favor, retreine usando model_train.py.")
            return None # Considera como não carregado se não estiver treinado
            
    except Exception as e:
        # Captura exceções que podem ocorrer durante o SQLIDetector.load_model se não forem tratadas internamente
        print(f"API ERRO: Exceção ao tentar carregar o modelo de '{caminho_modelo_para_carregar}': {e}")
        return None
    

def loadLastModel(info_file_path: str, models_dir: str) -> tuple[str | None, str | None]:
    """
    Lê o arquivo que contém o nome do último modelo treinado.

    """

    
    nome_arquivo_modelo = None
    caminho_completo_modelo = None

    if not os.path.exists(info_file_path):
        print(f"API AVISO: Arquivo de informação do último modelo ('{info_file_path}') não encontrado.")
        return nome_arquivo_modelo, caminho_completo_modelo

    try:
        with open(info_file_path, "r") as f:
            nome_arquivo_modelo = f.read().strip()
        
        if not nome_arquivo_modelo:
            print(f"API AVISO: Arquivo '{info_file_path}' está vazio.")
            return None, None # Retorna None para ambos se o arquivo estiver vazio
        
        caminho_completo_modelo = os.path.join(models_dir, nome_arquivo_modelo)
        print(f"API: Informação do último modelo encontrada: '{nome_arquivo_modelo}'")
        return nome_arquivo_modelo, caminho_completo_modelo
    
    except Exception as e:
        print(f"API AVISO: Erro ao ler '{info_file_path}': {e}")
        return None, None # Retorna None para ambos em caso de erro de leitura