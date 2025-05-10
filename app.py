from fastapi import FastAPI, Form, Depends, Query, Request
from sqlalchemy import text
from database import FormData, SessionLocal
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

app.mount("/assets", StaticFiles(directory="assets"), name="assets")


@app.get("/")
def root():
    return FileResponse("index.html")

@app.post("/submit-form")
async def submit_form(
    json: Request,
    db=Depends(get_db)
):
    data = await json.json()
    query = f"""
    INSERT INTO form (nome, email, cpf, endereco, date)
    VALUES ('{data.get('name')}', '{data.get('email')}', '{data.get('cpf')}', '{data.get('address')}', '{datetime.now()}')
    """
    db.execute(text(query))
    db.commit()
    return {"status": "success", "data": {"name": data.get('name')}}

@app.get("/search")
def search(name: str = Query(""), db=Depends(get_db)):
    query = f"SELECT * FROM form WHERE name LIKE '%{name}%'"
    result = db.execute(text(query))
    return [dict(row._mapping) for row in result]