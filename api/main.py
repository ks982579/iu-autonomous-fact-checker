"""
FastAPI application for fact-checking system
"""
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import random

# Import our models
from .models import (
    ClaimRequest, 
    FactCheckResponse, 
    PoliticalCheckResponse, 
    ClaimExtraction,
    PoliticalClassification
)

# Import existing claim processing components
from aieng.claim_extractor import ClaimExtractor
from aieng.claim_normalizer.normalizer import MyTextPreProcessor, KeywordExtractor


app = FastAPI(
    title="Fact Checking API", 
    description="API for processing and fact-checking claims",
    version="0.1.0"
)

# Add CORS middleware for browser extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your extension's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize models (you might want to do this lazily in production)
claim_model = None
text_processor = None
keyword_extractor = None


def get_claim_model():
    """Lazy loading of the claim extraction model"""
    global claim_model
    if claim_model is None:
        claim_model = ClaimExtractor()
    return claim_model


def get_text_processor():
    """Lazy loading of the text processor"""
    global text_processor
    if text_processor is None:
        text_processor = MyTextPreProcessor()
    return text_processor


def get_keyword_extractor():
    """Lazy loading of the keyword extractor"""
    global keyword_extractor
    if keyword_extractor is None:
        keyword_extractor = KeywordExtractor()
    return keyword_extractor


def fake_political_classifier(text: str) -> PoliticalCheckResponse:
    """
    Fake political content classifier - always returns non-political for now.
    Replace this with actual political content detection model later.
    """
    # For now, return non-political with high confidence
    return PoliticalCheckResponse(
        is_political=False,
        classification=PoliticalClassification.NON_POLITICAL,
        confidence=0.95
    )


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Fact Checking API is running"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}


@app.post("/check-political", response_model=PoliticalCheckResponse)
async def check_political_content(request: ClaimRequest):
    """
    Check if the provided text contains political content.
    Currently returns fake response - implement actual classifier later.
    """
    try:
        # For now, use fake classifier
        result = fake_political_classifier(request.text)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking political content: {str(e)}")


@app.post("/fact-check", response_model=FactCheckResponse)
async def fact_check_claim(request: ClaimRequest):
    """
    Main endpoint to process and fact-check claims.
    This endpoint follows the pipeline from main.py but returns results via API.
    """
    start_time = time.time()
    
    try:
        # Step 1: Check if content is political
        political_check = fake_political_classifier(request.text)
        
        # Step 2: Process text into sentences
        tp = get_text_processor()
        sentences = tp.get_sentences(request.text)
        
        # Step 3: Extract claims from sentences
        claim_extractor = get_claim_model()
        keyword_extractor = get_keyword_extractor()
        
        extracted_claims = []
        
        for sentence in sentences:
            claim_scores = claim_extractor.assign_claim_score_to_text(sentence)
            
            if claim_scores and len(claim_scores) > 0:
                score_data = claim_scores[0]
                label = score_data.get('label', '').lower()
                confidence = score_data.get('score', 0.0)
                
                if label == 'claim':
                    # Extract keywords for this claim
                    keywords = keyword_extractor.extract_keywords(sentence)
                    
                    extracted_claims.append(ClaimExtraction(
                        text=sentence,
                        confidence=confidence,
                        is_factual_claim=True
                    ))
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        
        # For now, we don't run the full RAG pipeline - just return claim extraction results
        return FactCheckResponse(
            original_text=request.text,
            is_political=political_check.is_political,
            extracted_claims=extracted_claims,
            fact_check_results=None,  # TODO: Implement full fact-checking pipeline
            processing_time_ms=processing_time,
            success=True
        )
        
    except Exception as e:
        processing_time = int((time.time() - start_time) * 1000)
        return FactCheckResponse(
            original_text=request.text,
            is_political=False,
            extracted_claims=[],
            fact_check_results=None,
            processing_time_ms=processing_time,
            success=False,
            error_message=str(e)
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)