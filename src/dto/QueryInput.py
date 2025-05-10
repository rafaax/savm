from pydantic import BaseModel

class QueryInputDTO(BaseModel):
    query: str
