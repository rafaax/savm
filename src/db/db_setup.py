import os
import sys
from sqlalchemy import JSON, Column, Integer, String, DateTime, Boolean, Text, Float,  create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


DATABASE_URL = "sqlite:///./database/sqlite.db"

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
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone(timedelta(hours=-3))).astimezone().strftime('%d/%m/%Y %H:%M:%S'), index=True)
    query_text = Column(String) 
    is_malicious_prediction = Column(Boolean)
    prediction_label = Column(Integer)
    probability_benign = Column(Float)
    probability_malicious = Column(Float)
    active_features = Column(JSON)

    def __repr__(self):
        return (f"<SQLiDetectionLog(id={self.id}, query='{self.query_text[:30]}...', "
                f"malicious={self.is_malicious_prediction}, label={self.prediction_label}, "
                f"prob_benign={self.probability_benign:.4f}, prob_malicious={self.probability_malicious:.4f})>")


class TrainedModelLog(Base):
    __tablename__ = "trained_model_logs"

    id = Column(Integer, primary_key=True, index=True)
    training_timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    model_filename = Column(String, unique=True, index=True)
    model_path = Column(String) 
    dataset_used_path = Column(String, nullable=True)
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)   
    false_negatives_count = Column(Integer, nullable=True)
    false_negatives_report_path = Column(String, nullable=True)
    training_duration_seconds = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)
    model_params = Column(Text, nullable=True) 
    feature_importance_path = Column(String)
    feature_importance_plot_path = Column(String)
    feature_importance_data = Column(JSON)

    def __repr__(self):
        return (f"<TrainedModelLog(id={self.id}, filename='{self.model_filename}', "
                f"accuracy={self.accuracy:.4f if self.accuracy is not None else 'N/A'}, "
                f"f1_score={self.f1_score:.4f if self.f1_score is not None else 'N/A'}, "
                f"training_duration_seconds={self.training_duration_seconds:.2f if self.training_duration_seconds is not None else 'N/A'})>")

Base.metadata.create_all(bind=engine)

print(f"Banco de dados SQLite inicializado em: {DATABASE_URL}")
print(f"Tabelas configuradas: {', '.join(Base.metadata.tables.keys())}")