import os
import traceback
import uvicorn
from fastapi import FastAPI, Query, Request, HTTPException, Form, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from pydantic import BaseModel
from database import FormData, SessionLocal, SQLiDetectionLog, get_db
from datetime import datetime
from src.sqli_detector import SQLIDetector
from dto.QueryInput import QueryInputDTO
from dto.DetectionResponse import DetectionResponseDTO
from src.utils.loaders import loadLastModel, loadModelSqli

MODELS_DIR = 'models'
LATEST_MODEL_INFO_FILE = os.path.join(MODELS_DIR, 'latest_model_info.txt')
MODEL_FILEPATH_TO_LOAD = None
sqli_detector_instance = None 


# busca o ultimo modelo treinado e pega o caminho dele e o nome dele
last_model, last_model_path = loadLastModel(LATEST_MODEL_INFO_FILE, MODELS_DIR) 

if last_model_path: # caso encontrou algum modelo ele instancia o sqli detector com ele
    sqli_detector_instance = loadModelSqli(last_model_path)

if not sqli_detector_instance:
    print(f"API ERRO: Nenhum modelo SQLi pôde ser carregado ou validado. "
          f"Verifique os avisos/erros anteriores. "
          f"Execute model_train.py para criar/atualizar o modelo e o arquivo '{LATEST_MODEL_INFO_FILE}'.")

app = FastAPI(
    title="API de Detecção de SQL Injection e Formulários",
    description="API para detectar queries SQL maliciosas e gerenciar dados de formulários.",
    version="1.0.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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



@app.post("/detect-sqli", response_model=DetectionResponseDTO, tags=["Detecção SQLi"])
async def detect_sqli_endpoint(payload: QueryInputDTO, db = Depends(get_db)):
    """
    Detecta se uma query é maliciosa ou não.
    """

    global sqli_detector_instance

    if not sqli_detector_instance or not sqli_detector_instance.is_trained():
        # Service Unavailable
        raise HTTPException(status_code=503, detail=f"Serviço de detecção de SQLi indisponível. Modelo não está treinado ou não foi carregado. ")
    
    try:
        prediction = sqli_detector_instance.predict_single(payload.query)
        is_malicious = bool(prediction == 1)

        log_entry = SQLiDetectionLog(
            query_text=payload.query,
            is_malicious_prediction=is_malicious,
            prediction_label=prediction
        )
        db.add(log_entry)
        db.commit()

        return DetectionResponseDTO(
            query=payload.query,
            is_malicious=is_malicious,
            prediction_label=prediction
        )
    
    except RuntimeError as e: # Captura "Modelo não treinado" se is_trained() falhar por algum motivo
        try:
            log_failure_entry = SQLiDetectionLog(
                query_text=payload.query,
                is_malicious_prediction=None,
                prediction_label=-2 # Código para erro de modelo
            )
            db.add(log_failure_entry)
            db.commit()
        except Exception as log_err:
            print(f"API ERRO: Falha ao logar erro de runtime na detecção: {log_err}")
            db.rollback()
        raise HTTPException(status_code=503, detail=f"Erro interno do modelo: {str(e)}")
    
    except Exception as e:
        try:
            log_exception_entry = SQLiDetectionLog(
                query_text=payload.query,
                is_malicious_prediction=None,
                prediction_label=-3 # Código para exceção geral
            )
            db.add(log_exception_entry)
            db.commit()
        except Exception as log_err:
            print(f"API ERRO: Falha ao logar exceção na detecção: {log_err}")
            db.rollback()

        print(f"API: Erro durante a predição: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Erro interno durante a predição: {str(e)}")


if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    os.makedirs('assets', exist_ok=True)
    os.makedirs('mocks', exist_ok=True)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)