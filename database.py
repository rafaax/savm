import os
import sys
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float,  create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv # Importar dotenv

load_dotenv()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("ERRO FATAL: A variável de ambiente 'DATABASE_URL' não está definida.")
    sys.exit(1)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False) 
Base = declarative_base()

class FormData(Base):
    __tablename__ = "form" 
    id = Column(Integer, primary_key=True, index=True)
    nome = Column(String, index=True) 
    email = Column(String, index=True)
    cpf = Column(String, unique=True, index=True) 
    endereco = Column(String) 
    date =  Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def __repr__(self): # para melhor debug -> nao fica aparecendo o endereço de memoria
        return f"<FormData(id={self.id}, nome='{self.nome}', email='{self.email}')>"
    


class SQLiDetectionLog(Base):
    __tablename__ = "sqli_detection_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    query_text = Column(String) 
    is_malicious_prediction = Column(Boolean)
    prediction_label = Column(Integer) # 0 para queries nao maliciosas, 1 para queries maliciosas

    def __repr__(self): # para melhor debug  -> nao fica aparecendo o endereço de memoria
        return f"<SQLiDetectionLog(id={self.id}, query='{self.query_text[:50]}...', malicious={self.is_malicious_prediction})>"


class TrainedModelLog(Base):
    __tablename__ = "trained_model_logs"

    id = Column(Integer, primary_key=True, index=True)
    training_timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    model_filename = Column(String, unique=True, index=True)
    model_path = Column(String) 
    dataset_used_path = Column(String, nullable=True)
    
    # Métricas de treinamento
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)   

    false_negatives_count = Column(Integer, nullable=True)
    false_negatives_report_path = Column(String, nullable=True)

    notes = Column(Text, nullable=True)

    def __repr__(self):
        return (f"<TrainedModelLog(id={self.id}, filename='{self.model_filename}', "
                f"accuracy={self.accuracy:.4f if self.accuracy is not None else 'N/A'}, "
                f"f1_score={self.f1_score:.4f if self.f1_score is not None else 'N/A'})>") # __repr__ atualizado


Base.metadata.create_all(bind=engine)

print(f"Banco de dados SQLite inicializado em: {DATABASE_URL}")
print(f"Tabelas configuradas: {', '.join(Base.metadata.tables.keys())}")