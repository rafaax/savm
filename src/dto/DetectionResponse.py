from pydantic import BaseModel
from typing import Dict, Any

class DetectionResponseDTO(BaseModel):
    query: str
    is_malicious: bool
    probability_benign: float
    probability_malicious: float
    label: int
    features_scaled: Dict[str, Any] 
    features_unscaled: Dict[str, Any]