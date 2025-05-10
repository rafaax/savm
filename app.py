from fastapi import FastAPI, Query, Request, HTTPException, Form, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import os
from database import FormData, SessionLocal
from datetime import datetime

import traceback
import uvicorn
from src.sqli_detector import SQLIDetector
from src.analyzer import ResultAnalyzer 

try:
    from database import SessionLocal, FormData # FormData é importado mas não usado diretamente nas rotas SQL raw
    print("API: SessionLocal e FormData importados de database.py com sucesso.")
except ImportError as e:
    print(f"API ERRO CRÍTICO: Falha ao importar de 'database.py': {e}. "
          "As rotas de banco de dados NÃO funcionarão. "
          "Certifique-se de que database.py está correto e no PYTHONPATH.")
    # Placeholders para permitir que a API inicie, mas as rotas de DB falharão claramente.
    class MockSession:
        def execute(self, *args, **kwargs): raise RuntimeError("DB não configurado: execute falhou")
        def commit(self): raise RuntimeError("DB não configurado: commit falhou")
        def close(self): pass
    def SessionLocal(): return MockSession()

MODEL_FILEPATH = 'models/sqli_detector_model.joblib' # Caminho onde o modelo treinado DEVE estar

# --- Carregar o Modelo SQLi na Inicialização ---
print(f"API: Tentando carregar o modelo SQLi de: {MODEL_FILEPATH}")
sqli_detector_instance = SQLIDetector.load_model(MODEL_FILEPATH)

if sqli_detector_instance:
    if sqli_detector_instance.is_trained():
        print("API: Modelo SQLi pré-treinado carregado com sucesso.")
    else:
        # Isso é um problema, pois a API não terá como treinar.
        print("API ERRO: Modelo carregado de '{MODEL_FILEPATH}', mas não está marcado como treinado! "
              "Por favor, retreine usando model_train.py.")
        # Tratar como não carregado efetivamente
        sqli_detector_instance = None 
else:
    print(f"API ERRO: Nenhum modelo pré-treinado encontrado em '{MODEL_FILEPATH}' ou falha ao carregar. "
          "A API não poderá fazer detecções de SQLi até que um modelo válido seja fornecido. "
          "Execute model_train.py para criar o arquivo do modelo.")
    # sqli_detector_instance já será None

app = FastAPI(
    title="API de Detecção de SQL Injection e Formulários",
    description="API para detectar queries SQL maliciosas e gerenciar dados de formulários.",
    version="1.0.1"
)

# --- Middleware CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Restrinja em produção
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Modelos Pydantic ---
class QueryInput(BaseModel):
    query: str

class DetectionResponse(BaseModel):
    query: str
    is_malicious: bool
    prediction_label: int

# (Se você tiver o endpoint de análise de Falsos Negativos, mantenha os modelos Pydantic para ele)
class CommonPattern(BaseModel):
    pattern: str
    count: int
class FalseNegativeAnalysisDetails(BaseModel):
    length_statistics: Optional[Dict[str, float]] = None
    common_patterns: Optional[List[CommonPattern]] = None
class FalseNegativeRecord(BaseModel):
    query: str
    label: int
class FalseNegativesResponse(BaseModel):
    message: str
    count: int
    false_negatives: List[FalseNegativeRecord] = []
    analysis_details: Optional[FalseNegativeAnalysisDetails] = None
    file_path: Optional[str] = None


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Endpoints ---



try:
    app.mount("/assets", StaticFiles(directory="assets"), name="assets")
    print("API: Diretório estático '/assets' montado.")
except RuntimeError as e:
    print(f"API AVISO: Não foi possível montar o diretório estático 'assets': {e}. ")

# --- Rotas Originais da sua Interface (JÁ ALINHADAS COM database.py MODIFICADO) ---
@app.get("/", tags=["Interface"])
async def root():
    if not os.path.exists("index.html"):
        raise HTTPException(status_code=404, detail="Arquivo index.html não encontrado.")
    return FileResponse("index.html")

@app.post("/submit-form", tags=["Formulário"])
async def submit_form_endpoint(
    request_data: Request,
    db=Depends(get_db)
):
    data = await request_data.json()
    # A query SQL já espera 'nome', 'email', 'cpf', 'endereco', 'date'
    query_sql = text("""
    INSERT INTO form (nome, email, cpf, endereco, date)
    VALUES (:name, :email, :cpf, :address, :date)
    """) # Os placeholders :name, :email etc. são os nomes das chaves no dicionário de parâmetros
    try:
        db.execute(query_sql, {
            "name": data.get('name'),       # Chave 'name' no JSON -> placeholder :name
            "email": data.get('email'),     # Chave 'email' no JSON -> placeholder :email
            "cpf": data.get('cpf'),         # Chave 'cpf' no JSON -> placeholder :cpf
            "address": data.get('address'), # Chave 'address' no JSON -> placeholder :address
            "date": datetime.now()          # Valor gerado -> placeholder :date
        })
        db.commit()
        return {"status": "success", "data": {"name": data.get('name')}} # Retorna o 'name' do JSON
    except Exception as e:
        import traceback
        print(f"API ERRO ao submeter formulário: {e}\n{traceback.format_exc()}")
        db.rollback() # Importante adicionar rollback em caso de erro na transação
        raise HTTPException(status_code=500, detail=f"Erro ao processar o formulário: {str(e)}")

@app.get("/search", tags=["Formulário"])
async def search_endpoint(name: str = Query(""), db=Depends(get_db)):
    # A query SQL busca na coluna 'nome'
    query_sql = text("SELECT * FROM form WHERE nome LIKE :name_pattern") # Coluna 'nome'
    try:
        result = db.execute(query_sql, {"name_pattern": f"%{name}%"})
        return [dict(row._mapping) for row in result]
    except Exception as e:
        import traceback
        print(f"API ERRO ao buscar: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Erro ao realizar a busca: {str(e)}")



@app.post("/detect-sqli", response_model=DetectionResponse, tags=["Detecção SQLi"])
async def detect_sqli_endpoint(payload: QueryInput):
    """
    Detecta se uma query SQL fornecida é maliciosa.
    Requer um modelo pré-treinado (`models/sqli_detector_model.joblib`).
    """
    global sqli_detector_instance # Acessa a instância carregada na inicialização

    if not sqli_detector_instance or not sqli_detector_instance.is_trained():
        raise HTTPException(
            status_code=503, # Service Unavailable
            detail=f"Serviço de detecção de SQLi indisponível. Modelo não está treinado ou não foi carregado. "
                   f"Verifique se '{MODEL_FILEPATH}' existe e foi gerado por model_train.py."
        )
    
    try:
        prediction = sqli_detector_instance.predict_single(payload.query)
        is_malicious = bool(prediction == 1)
        return DetectionResponse(
            query=payload.query,
            is_malicious=is_malicious,
            prediction_label=prediction
        )
    except RuntimeError as e: # Captura "Modelo não treinado" se is_trained() falhar por algum motivo
        # Isso não deveria acontecer se a verificação acima passou, mas é uma salvaguarda.
        raise HTTPException(status_code=503, detail=f"Erro interno do modelo: {str(e)}")
    except Exception as e:
        import traceback
        print(f"API: Erro durante a predição: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Erro interno durante a predição: {str(e)}")


# Se você ainda quiser o endpoint de análise de falsos negativos do último treinamento
# (lembre-se que ele usará os dados cacheados DENTRO do arquivo .joblib)
@app.get("/analyze-last-training-false-negatives", response_model=FalseNegativesResponse, tags=["Análise do Modelo"])
async def analyze_last_training_fn_endpoint():
    """
    Analisa os falsos negativos do conjunto de teste usado durante o último treinamento
    do modelo carregado (`models/sqli_detector_model.joblib`).
    """
    global sqli_detector_instance
    
    if not sqli_detector_instance or not sqli_detector_instance.is_trained():
        raise HTTPException(status_code=503, 
                            detail="Serviço de análise indisponível. Modelo não treinado ou não carregado.")

    # get_last_evaluation_data() deve retornar os dados que foram salvos com o modelo
    eval_data = sqli_detector_instance.get_last_evaluation_data()
    if eval_data is None:
        raise HTTPException(status_code=404, 
                            detail="Nenhum dado de avaliação do último treinamento disponível no modelo carregado.")

    df_cache, y_test_cache, y_pred_cache, test_indices_cache = eval_data
    
    analyzer = ResultAnalyzer() # Crie uma instância
    try:
        fn_df, analysis_details_dict = analyzer.analyze_false_negatives(
            df_cache, y_test_cache, y_pred_cache, test_indices_cache
        )
        
        output_file_path = 'results/api_loaded_model_false_negatives.csv' # Salvo pela API
        fn_records = []
        analysis_response_payload = None

        if analysis_details_dict:
            analysis_response_payload = FalseNegativeAnalysisDetails(
                length_statistics=analysis_details_dict.get('length_statistics'),
                common_patterns=analysis_details_dict.get('common_patterns')
            )

        if fn_df is not None and not fn_df.empty:
            os.makedirs('results', exist_ok=True)
            fn_df.to_csv(output_file_path, index=False)
            
            for _, row in fn_df.iterrows():
                fn_records.append(FalseNegativeRecord(
                    query=row.get('query', 'N/A'), 
                    label=int(row.get('label', -1)) 
                ))
            
            return FalseNegativesResponse(
                message=f"Análise de falsos negativos do modelo carregado concluída. {len(fn_df)} registros salvos.",
                count=len(fn_df),
                false_negatives=fn_records,
                analysis_details=analysis_response_payload,
                file_path=output_file_path
            )
        else:
            return FalseNegativesResponse(
                message="Nenhum falso negativo encontrado nos dados de avaliação do modelo carregado.",
                count=0,
                false_negatives=[],
                analysis_details=analysis_response_payload
            )
    except Exception as e:
        print(f"API: Erro durante a análise de falsos negativos do modelo carregado: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Erro interno durante a análise: {str(e)}")


# Adicione suas outras rotas (como /submit-form, /search) aqui, se necessário.
# Elas não dependem do modelo de ML.

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True) # Para os CSVs de análise
    os.makedirs('assets', exist_ok=True)
    os.makedirs('mocks', exist_ok=True) # Para o dataset se model_train.py precisar dele
    
    print("Verifique se 'models/sqli_detector_model.joblib' existe ANTES de iniciar a API.")
    print("Execute 'python model_train.py' para criar/atualizar o arquivo do modelo.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
