from pydantic import BaseModel

class DetectionResponseDTO(BaseModel):
    query: str
    is_malicious: bool
    probability_benign: float
    probability_malicious: float
    label: int