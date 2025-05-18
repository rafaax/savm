import os
import uvicorn
from fastapi import FastAPI, Query, Form, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from database import FormData, SessionLocal, SQLiDetectionLog, get_db
from src.utils.loaders import loadLastModel, loadModelSqli
from api.v1.routes import router as api_router

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

app.include_router(api_router)

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    os.makedirs('assets', exist_ok=True)
    os.makedirs('mocks', exist_ok=True)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)