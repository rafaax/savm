from pydantic import BaseModel

class DetectionResponseDTO(BaseModel):
    query: str
    is_malicious: bool
    prediction_label: int