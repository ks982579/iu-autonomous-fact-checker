"""
Pydantic models for API request and response schemas
"""
from pydantic import BaseModel
from typing import List, Optional
from enum import Enum


class PoliticalClassification(str, Enum):
    POLITICAL = "political"
    NON_POLITICAL = "non_political"


class ClaimRequest(BaseModel):
    """Request model for claim fact-checking"""
    text: str
    

class ClaimExtraction(BaseModel):
    """Individual claim with confidence score"""
    text: str
    confidence: float
    is_factual_claim: bool


class PoliticalCheckResponse(BaseModel):
    """Response for political content classification"""
    is_political: bool
    classification: PoliticalClassification
    confidence: float


class FactCheckResponse(BaseModel):
    """Response model for fact-checking results"""
    original_text: str
    is_political: bool
    extracted_claims: List[ClaimExtraction]
    fact_check_results: Optional[List[dict]] = None
    processing_time_ms: int
    success: bool
    error_message: Optional[str] = None