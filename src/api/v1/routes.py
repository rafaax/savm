from fastapi import APIRouter, HTTPException, Request, Depends
from sqlalchemy import text
from fastapi.responses import FileResponse
from datetime import datetime, timezone, timedelta
import os
import traceback
from db.db_setup import get_db
from dto.QueryInput import QueryInputDTO
from dto.DetectionResponse import DetectionResponseDTO
from db.db_setup import SQLiDetectionLog
from utils.loaders import loadLastModel, loadModelSqli

MODELS_DIR = 'models'
LATEST_MODEL_INFO_FILE = os.path.join(MODELS_DIR, 'latest_model_info.txt')
MODEL_FILEPATH_TO_LOAD = None
sqli_detector_instance = None 

print(f"LATEST_MODEL_INFO_FILE: {LATEST_MODEL_INFO_FILE}")
print(f"MODELS_DIR: {MODELS_DIR}")


# busca o ultimo modelo treinado e pega o caminho dele e o nome dele
last_model, last_model_path = loadLastModel(LATEST_MODEL_INFO_FILE, MODELS_DIR) 

if last_model_path: # caso encontrou algum modelo ele instancia o sqli detector com ele
    sqli_detector_instance = loadModelSqli(last_model_path)

if not sqli_detector_instance:
    print(f"API ERRO: Nenhum modelo SQLi pôde ser carregado ou validado. "
          f"Verifique os avisos/erros anteriores. "
          f"Execute model_train.py para criar/atualizar o modelo e o arquivo '{LATEST_MODEL_INFO_FILE}'.")

router = APIRouter()

@router.get("/", tags=["Interface"])
async def root():
    if not os.path.exists("index.html"):
        raise HTTPException(status_code=404, detail="Arquivo index.html não encontrado.")
    
    return FileResponse("index.html")

@router.post("/submit-form", tags=["Formulário"])
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
            "date": datetime.now(timezone(timedelta(hours=-3))).astimezone().strftime('%d/%m/%Y %H:%M:%S')
        })

        db.commit()

        return {"status": "success", "data": {"name": data.get('name')}}
    except Exception as e:
        print(f"API ERRO ao submeter formulário: {e}\n{traceback.format_exc()}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Erro ao processar o formulário: {str(e)}")

@router.post("/detect-sqli", response_model=DetectionResponseDTO, tags=["Detecção SQLi"])
async def detect_sqli_endpoint(payload: QueryInputDTO, db=Depends(get_db)):
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
            probability_malicious=float(prediction["probability_malicious"]),
            active_features=prediction['active_features']
        )
        db.add(new_log_entry)
        db.commit()

        return prediction

    except RuntimeError as e:
        print(f"API ALERTA: RuntimeError durante a predição do modelo: {e}\n{traceback.format_exc()}")
        db.rollback()
        raise HTTPException(status_code=503, detail=f"Erro no serviço de detecção: {str(e)}")

    except Exception as e:
        print(f"API ERRO: Exceção inesperada durante a predição ou log inicial: {e}\n{traceback.format_exc()}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Erro interno do servidor durante o processamento da sua requisição.")

@router.get("/trained-model-logs", tags=["Trained Model Logs"])
async def get_trained_model_logs(db=Depends(get_db)):
    try:
        result = db.execute(text("SELECT * FROM trained_model_logs"))
        return [dict(row._mapping) for row in result]
    
    except Exception as e:
        print(f"API ERRO ao buscar logs de modelo: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Erro ao buscar logs de modelo.")

@router.get("/all-queries-detected", tags=["All Queries"])
async def get_all_queries(db=Depends(get_db)):
    try:
        result = db.execute(text("SELECT * FROM sqli_detection_logs"))
        return [dict(row._mapping) for row in result]
    
    except Exception as e:
        print(f"API ERRO ao buscar as queries: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Erro ao buscar as queries executadas no sistema.")
    
@router.get("/all-users-registred", tags=["All Users"])
async def get_all_users(db=Depends(get_db)):
    try:
        result = db.execute(text("SELECT * FROM form"))
        return [dict(row._mapping) for row in result]
    
    except Exception as e:
        print(f"API ERRO ao buscar os usuarios: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Erro ao buscar usuarios no sistema.")