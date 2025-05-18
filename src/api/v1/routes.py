from fastapi import APIRouter, HTTPException, Query, Request, Depends
from sqlalchemy import text
from fastapi.responses import FileResponse
from datetime import datetime
import os
import traceback
from database import get_db
from dto.QueryInput import QueryInputDTO
from dto.DetectionResponse import DetectionResponseDTO
from database import SQLiDetectionLog

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
            "date": datetime.now()
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
            probability_malicious=float(prediction["probability_malicious"])
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