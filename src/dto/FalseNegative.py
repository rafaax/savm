from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from dto.CommonPatern import CommonPatternDTO


class FalseNegativeAnalysisDetailsDTO(BaseModel):
    length_statistics: Optional[Dict[str, float]] = None
    common_patterns: Optional[List[CommonPatternDTO]] = None

class FalseNegativeRecordDTO(BaseModel):
    query: str
    label: int

class FalseNegativesResponseDTO(BaseModel):
    message: str
    count: int
    false_negatives: List[FalseNegativeRecordDTO] = []
    analysis_details: Optional[FalseNegativeAnalysisDetailsDTO] = None
    file_path: Optional[str] = None