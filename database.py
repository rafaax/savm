import os
from sqlalchemy import Column, Integer, String, DateTime, create_engine # Adicionado DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pathlib import Path
from datetime import datetime 
db_dir = Path("database")
db_dir.mkdir(parents=True, exist_ok=True)

db_path = db_dir / "sqlite.db"
DATABASE_URL = f"sqlite:///{db_path.resolve()}" # Usar caminho absoluto resolvido

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False) # autocommit=False é default
Base = declarative_base()

class FormData(Base):
    __tablename__ = "form" # Nome da tabela é 'form'

    id = Column(Integer, primary_key=True, index=True)
    nome = Column(String, index=True) # Era 'name', mudado para 'nome'
    email = Column(String, index=True) # Adicionado index=True para email, pode ser útil
    cpf = Column(String, unique=True, index=True) # Adicionado CPF, geralmente é único
    endereco = Column(String) # Adicionado Endereço
    date = Column(DateTime, default=datetime.utcnow) # Adicionado campo de data com valor default

    def __repr__(self):
        return f"<FormData(id={self.id}, nome='{self.nome}', email='{self.email}')>"

Base.metadata.create_all(bind=engine)

print(f"Banco de dados SQLite inicializado em: {DATABASE_URL}")
print("Tabela 'FormData' (form) configurada com as colunas: id, nome, email, cpf, endereco, date")