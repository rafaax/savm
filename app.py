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
        raise HTTPException(status_code=503, detail="Serviço de detecção de SQLi indisponível. Modelo não treinado ou não carregado.")

    try:
        prediction = sqli_detector_instance.predict_single(payload.query)
        
        new_log_entry = SQLiDetectionLog(
            query_text=prediction["query"],
            is_malicious_prediction=prediction["is_malicious"],
            prediction_label=prediction["label"],
            probability_benign=float(prediction["probability_benign"]),
            probability_malicious=float(prediction["probability_malicious"])
        )
        db.add(new_log_entry)
        db.commit()

        return prediction

    except RuntimeError as e: # Captura erros específicos do modelo, como "Modelo não treinado" durante a predição
        print(f"API ALERTA: RuntimeError durante a predição do modelo: {e}\n{traceback.format_exc()}")
        try:
            db.rollback() # Garante que a sessão está limpa antes de tentar nova transação
        except Exception as log_err:
            print(f"API ERRO CRÍTICO: Falha ao logar erro de runtime na detecção (após rollback): {log_err}")
            try:
                db.rollback()
            except Exception:
                pass
        raise HTTPException(status_code=503, detail=f"Erro no serviço de detecção: {str(e)}")

    except Exception as e:
        print(f"API ERRO: Exceção inesperada durante a predição ou log inicial: {e}\n{traceback.format_exc()}")
        
        try:
            db.rollback()
        except Exception as log_err:
            print(f"API ERRO CRÍTICO: Falha ao logar exceção geral na detecção (após rollback): {log_err}")
            try:
                db.rollback()
            except Exception:
                pass
        raise HTTPException(status_code=500, detail="Erro interno do servidor durante o processamento da sua requisição.")


if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    os.makedirs('assets', exist_ok=True)
    os.makedirs('mocks', exist_ok=True)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)