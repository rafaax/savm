from pydantic import BaseModel

class CommonPatternDTO(BaseModel):
    pattern: str
    count: int