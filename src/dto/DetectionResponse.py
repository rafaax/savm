from pydantic import BaseModel
from typing import Dict, Any, Optional, Union

class DetectionResponseDTO(BaseModel):
    query: str
    is_malicious: bool
    probability_benign: float
    probability_malicious: float
    label: int
    active_features: Dict[str, Union[int, float, bool]]