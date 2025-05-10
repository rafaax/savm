import os
import traceback
import uvicorn

from fastapi import FastAPI, Query, Request, HTTPException, Form, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from pydantic import BaseModel
from database import FormData, SessionLocal
from datetime import datetime
from src.sqli_detector import SQLIDetector
from dto.QueryInput import QueryInputDTO

MODELS_DIR = 'models' # Defina o diretório base dos modelos
LATEST_MODEL_INFO_FILE = os.path.join(MODELS_DIR, 'latest_model_info.txt')
MODEL_FILEPATH_TO_LOAD = None
sqli_detector_instance = None 

if os.path.exists(LATEST_MODEL_INFO_FILE):
    try:
        with open(LATEST_MODEL_INFO_FILE, "r") as f:
            latest_model_filename = f.read().strip()
        if latest_model_filename:
            MODEL_FILEPATH_TO_LOAD = os.path.join(MODELS_DIR, latest_model_filename)
            print(f"API: Informação do último modelo encontrada: '{latest_model_filename}'")
            print(f"API: Tentando carregar o modelo SQLi de: {MODEL_FILEPATH_TO_LOAD}")
            if os.path.exists(MODEL_FILEPATH_TO_LOAD):
                sqli_detector_instance = SQLIDetector.load_model(MODEL_FILEPATH_TO_LOAD)
            else:
                print(f"API ERRO: Arquivo do modelo '{MODEL_FILEPATH_TO_LOAD}' (indicado como o mais recente) não encontrado.")
        else:
            print(f"API AVISO: Arquivo '{LATEST_MODEL_INFO_FILE}' está vazio.")
    except Exception as e:
        print(f"API AVISO: Erro ao ler '{LATEST_MODEL_INFO_FILE}': {e}")
else:
    print(f"API AVISO: Arquivo de informação do último modelo ('{LATEST_MODEL_INFO_FILE}') não encontrado. Não é possível carregar um modelo específico.")

# Verifica se o modelo foi carregado e está treinado
if sqli_detector_instance:
    if sqli_detector_instance.is_trained():
        print("API: Modelo SQLi pré-treinado carregado com sucesso.")
    else:
        print(f"API ERRO: Modelo carregado de '{MODEL_FILEPATH_TO_LOAD}', mas não está marcado como treinado! "
              "Por favor, retreine usando model_train.py.")
        sqli_detector_instance = None # Considera como não carregado
elif MODEL_FILEPATH_TO_LOAD and not os.path.exists(MODEL_FILEPATH_TO_LOAD):
    # Se tentamos carregar um arquivo específico mas ele não existia (já logado acima).
    pass
else:
     print(f"API ERRO: Nenhum modelo SQLi pôde ser carregado. "
           "Execute model_train.py para criar um modelo e o arquivo '{LATEST_MODEL_INFO_FILE}'.")

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

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



app.mount("/assets", StaticFiles(directory="assets"), name="assets") # definindo a pasta assets para acessar pela tela do formulario

# --- Endpoints ---

@app.get("/", tags=["Interface"])
async def root():
    if not os.path.exists("index.html"):
        raise HTTPException(status_code=404, detail="Arquivo index.html não encontrado.")
    
    return FileResponse("index.html")

@app.post("/submit-form", tags=["Formulário"])
async def submit_form_endpoint(request_data: Request, db=Depends(get_db)):
    data = await request_data.json()

    # A query SQL já espera 'nome', 'email', 'cpf', 'endereco', 'date']
    query_sql = text("""
    INSERT INTO form (nome, email, cpf, endereco, date)
    VALUES (:name, :email, :cpf, :address, :date)
    """)
    
    try:
        db.execute(query_sql, {
            "name": data.get('name'),
            "email": data.get('email'),
            "cpf": data.get('cpf'),
            "address": data.get('address'),
            "date": datetime.now()
        })

        db.commit()

        return {"status": "success", "data": {"name": data.get('name')}}
    except Exception as e:
        print(f"API ERRO ao submeter formulário: {e}\n{traceback.format_exc()}")
        db.rollback() # rollback em caso de erro na transação
        raise HTTPException(status_code=500, detail=f"Erro ao processar o formulário: {str(e)}")


@app.get("/search", tags=["Formulário"])
async def search_endpoint(name: str = Query(""), db=Depends(get_db)):
    query_sql = text("SELECT * FROM form WHERE nome LIKE :name_pattern")
    try:
        result = db.execute(query_sql, {"name_pattern": f"%{name}%"})
        return [dict(row._mapping) for row in result]
    
    except Exception as e:
        print(f"API ERRO ao buscar: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Erro ao realizar a busca: {str(e)}")



@app.post("/detect-sqli", response_model=DetectionResponse, tags=["Detecção SQLi"])
async def detect_sqli_endpoint(payload: QueryInputDTO):
    """
    Detecta se uma query SQL fornecida é maliciosa.
    Requer um modelo pré-treinado (`models/sqli_detector_model.joblib`).
    """

    global sqli_detector_instance

    if not sqli_detector_instance or not sqli_detector_instance.is_trained():
        raise HTTPException(
            status_code=503, # Service Unavailable
            detail=f"Serviço de detecção de SQLi indisponível. Modelo não está treinado ou não foi carregado. "
                   f"Verifique se '{MODEL_FILEPATH_TO_LOAD}' existe e foi gerado por model_train.py."
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
        raise HTTPException(status_code=503, detail=f"Erro interno do modelo: {str(e)}")
    
    except Exception as e:
        print(f"API: Erro durante a predição: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Erro interno durante a predição: {str(e)}")


if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    os.makedirs('assets', exist_ok=True)
    os.makedirs('mocks', exist_ok=True)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)