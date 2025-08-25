"""
FastAPI application for fact-checking system
"""
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import random

# Import our models
from .models import (
    ClaimRequest,
    FactCheckResponse,
    PoliticalCheckResponse,
    ClaimExtraction,
    PoliticalClassification,
    FactCheckResult
)

# Import existing claim processing components
from aieng.claim_extractor import ClaimExtractor
from aieng.claim_normalizer.normalizer import MyTextPreProcessor, KeywordExtractor
from aieng.political_detector.political_classifier import PoliticalContentClassifier
from aieng.rag_system.rag_eng import *
from aieng.rag_system.vectordb import VectorPipeline
from aieng.judge_model import ClaimJudgeTextClassifier


def map_judge_label(label: str) -> str:
    """Map judge model labels to human-readable format"""
    label_mapping = {
        'LABEL_0': 'SUPPORTS',
        'LABEL_1': 'REFUTES',
        'LABEL_2': 'NOT_ENOUGH_INFO'
    }
    return label_mapping.get(label, label)


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
political_classifier = None
vector_pipeline = None
judge_model = None
allsides = None
gdelt = None
wiki = None


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


def get_political_classifier():
    """Lazy loading of the political content classifier"""
    global political_classifier
    if political_classifier is None:
        political_classifier = PoliticalContentClassifier(
            use_simple_fallback=False)
    return political_classifier


def get_vector_pipeline():
    """Lazy loading of the vector pipeline"""
    global vector_pipeline
    if vector_pipeline is None:
        vector_pipeline = VectorPipeline(chunk_size=64, overlap=8)
    return vector_pipeline


def get_judge_model():
    """Lazy loading of the judge model"""
    global judge_model
    if judge_model is None:
        judge_model = ClaimJudgeTextClassifier()
    return judge_model


def get_news_clients():
    """Lazy loading of news clients"""
    global allsides, gdelt, wiki
    if allsides is None:
        allsides = AllSidesNews()
    if gdelt is None:
        gdelt = GDELTClient()
    if wiki is None:
        wiki = WikiClient()
    return allsides, gdelt, wiki


def standardize_preprocessing(text):
    """
    Standard preprocessing for both TweetTopic and your political data
    """
    # Replace URLs
    text = re.sub(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '{{URL}}', text)

    # Replace all @mentions with {{USERNAME}}
    text = re.sub(r'@\w+', '{{USERNAME}}', text)

    return text


def classify_political_content(text: str) -> PoliticalCheckResponse:
    """
    Real political content classifier using trained model.
    """
    classifier = get_political_classifier()
    result = classifier.classify_content(text)

    # Map to API response format
    classification = (
        PoliticalClassification.POLITICAL if result['is_political']
        else PoliticalClassification.NON_POLITICAL
    )

    # If it is non-political but the confidence is low, we consider it political for testing
    if not result['is_political'] and result.get('confidence', 0) < 0.75:
        classification = PoliticalClassification.POLITICAL

    return PoliticalCheckResponse(
        is_political=result['is_political'],
        classification=classification,
        confidence=result['confidence']
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
    Check if the provided text contains political content using trained classifier.
    """
    try:
        result = classify_political_content(request.text)
        return result

    except Exception as e:
        return PoliticalCheckResponse(
            is_political=False,
            classification=PoliticalClassification.NON_POLITICAL,
            confidence=0.0,
            success=False,
            message=f"Error checking political content: {str(e)}"
        )


@app.post("/fact-check", response_model=FactCheckResponse)
async def fact_check_claim(request: ClaimRequest):
    """
    Main endpoint to process and fact-check claims.
    This endpoint follows the full pipeline from main.py.
    """
    start_time = time.time()

    try:
        text = request.text
        # Check if content is political
        political_check = classify_political_content(text)
        print(political_check)

        if political_check.classification != PoliticalClassification.POLITICAL:
            processing_time = int((time.time() - start_time) * 1000)
            # Ensure minimum processing time of 1ms to avoid 0
            processing_time = max(processing_time, 1)
            return FactCheckResponse(
                original_text=request.text,
                is_political=False,
                extracted_claims=[],
                fact_check_results=[],
                processing_time_ms=processing_time,
                success=False,
                message="Non political content detected."
            )

        # Process text into sentences
        tp = get_text_processor()
        sentences = tp.get_sentences(text)

        # Extract claims from sentences
        claim_extractor = get_claim_model()
        keyword_extractor = get_keyword_extractor()

        extracted_claims = []
        claims_text = []
        keywords_list = []

        for sentence in sentences:
            claim_scores = claim_extractor.assign_claim_score_to_text(sentence)

            if claim_scores and len(claim_scores) > 0:
                score_data = claim_scores[0]
                label = score_data.get('label', '').lower()
                confidence = score_data.get('confidence', 0.0)
                print(claim_scores)

                if label != 'claim' and confidence < .80:  # Require high confidence for non-claim
                    label = 'claim'

                if label == 'claim':  # High confidence threshold for claims
                    # Extract keywords for this claim
                    keywords = keyword_extractor.extract_keywords(sentence)

                    extracted_claims.append(ClaimExtraction(
                        text=sentence,
                        confidence=confidence,
                        is_factual_claim=True
                    ))
                    claims_text.append(sentence)
                    keywords_list.append(keywords)

        if not claims_text:
            processing_time = int((time.time() - start_time) * 1000)
            processing_time = max(processing_time, 1)
            return FactCheckResponse(
                original_text=request.text,
                is_political=True,
                extracted_claims=[],
                fact_check_results=[],
                processing_time_ms=processing_time,
                success=False,
                message="No factual claims detected."
            )

        # Step 4: Get news sources and update vector database
        vector_pipeline = get_vector_pipeline()
        # TODO: AllSides is not a good choice so it is no longer used
        _allsides, gdelt, wiki = get_news_clients()

        for keywords in keywords_list:
            # GDELT search
            keywords = list(set([x.casefold() for x in keywords]))
            try:
                gdelt_stories = gdelt.search_news(keywords[:7], max_records=5)
                if len(gdelt_stories) == 0:
                    gdelt_stories = gdelt.search_news(
                        keywords[:5], max_records=3)
                if len(gdelt_stories) == 0:
                    gdelt_stories = gdelt.search_news(
                        keywords[:2], max_records=3)
            
                for story in gdelt_stories:
                    print(story.get('title', "{{GDELT TITLE}}"))
                    if not vector_pipeline.article_exists(story.get("url")):
                        scraper = SimpleScraper(timeout=10, delay=1)
                        scraper_response = scraper.scrape_other_content(
                            story.get("url"))

                        if scraper_response.content:
                            print(f"ADDING: {str(story.get('url', ''))}")
                            vector_pipeline.add_article({
                                'url': str(story.get('url', '')),
                                'title': str(story.get('title', '')),
                                'source': str(story.get('source', '')),
                                'author': "GDELT Project",
                                'published_at': str(story.get('published_at', '')),
                                'full_content': scraper_response.content
                            })
            except Exception as e:
                print(f"GDELT search error: {e}")

            # NewsAPI search
            try:
                news_api_response = get_news(keywords[:5], page_size=3)
                news_api_json = news_api_response.json()
                articles = news_api_json.get("articles", [])

                if len(articles) == 0:
                    news_api_response = get_news(keywords[:3], page_size=5)
                    news_api_json = news_api_response.json()
                    articles = news_api_json.get("articles", [])

                print("NEWS API")
                print(articles)

                for article_md in articles:
                    metadata = NewsApiJsonResponseArticle(article_md)
                    print(metadata.title, "{{NewsAPI TITLE}}")

                    if not vector_pipeline.article_exists(metadata.url):
                        scraper = SimpleScraper(timeout=10, delay=1)
                        scraper_response = scraper.scrape_article_content(
                            metadata)

                        if scraper_response.content:
                            print(f"ADDING: {metadata.url}")
                            vector_pipeline.add_article({
                                'url': metadata.url,
                                'title': metadata.title,
                                'source': metadata.source.site_name,
                                'author': metadata.author,
                                'published_at': metadata.published_at,
                                'full_content': scraper_response.content
                            })
            except Exception as e:
                print(f"NewsAPI search error: {e}")

            # Wikipedia search
            try:
                wiki_result = wiki(keywords, vector_pipeline.article_exists)
                if wiki_result and wiki_result.get('success'):
                    vector_pipeline.add_article({
                        'url': str(wiki_result.get('url', '')),
                        'title': str(wiki_result.get('title', '')),
                        'source': "Wikipedia",
                        'author': "Wikipedia",
                        'published_at': str(wiki_result.get('published_at', '')),
                        'full_content': str(wiki_result.get('text_content')),
                    })
            except Exception as e:
                print(f"Wikipedia search error: {e}")

        # Step 5: Search for similar content and judge claims
        judge_model = get_judge_model()
        fact_check_results = []

        for claim in claims_text:
            try:
                search_results = vector_pipeline.search_similar_content(
                    query=claim,
                    n_results=5,
                )

                if search_results.get("success", False):
                    evidence = []
                    source_urls = []
                    for chunk in search_results.get('results', []):
                        ev = chunk.get('content')
                        if ev:
                            evidence.append(ev)

                        # Extract URL from metadata
                        metadata = chunk.get('metadata', {})
                        url = metadata.get('url')
                        if url and url not in source_urls:
                            source_urls.append(url)

                    # URLs collected from Evidence
                    if evidence:
                        print("JUDGING")
                        print(f"{claim}")
                        print(evidence)
                        # Pass all evidence as List[str]
                        verdict_result = judge_model(claim, evidence)
                        # Judge model returns a list with one dict like HF pipeline
                        verdict_dict = verdict_result[0] if isinstance(
                            verdict_result, list) and len(verdict_result) > 0 else {}
                        raw_label = verdict_dict.get('label', 'UNKNOWN')
                        human_label = map_judge_label(raw_label)
                        fact_check_results.append(FactCheckResult(
                            claim=claim,
                            verdict=human_label,
                            confidence=verdict_dict.get('score', 0.0),
                            evidence_count=len(evidence),
                            source_urls=source_urls
                        ))
                    else:
                        fact_check_results.append(FactCheckResult(
                            claim=claim,
                            verdict="INSUFFICIENT_EVIDENCE",
                            confidence=0.0,
                            evidence_count=0,
                            source_urls=source_urls
                        ))
                else:
                    fact_check_results.append(FactCheckResult(
                        claim=claim,
                        verdict="NO_EVIDENCE_FOUND",
                        confidence=0.0,
                        evidence_count=0,
                        source_urls=[]
                    ))
            except Exception as e:
                fact_check_results.append(FactCheckResult(
                    claim=claim,
                    verdict="ERROR",
                    confidence=0.0,
                    evidence_count=0,
                    source_urls=[]
                ))

        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)

        return FactCheckResponse(
            original_text=request.text,
            is_political=True,
            extracted_claims=extracted_claims,
            fact_check_results=fact_check_results,
            processing_time_ms=processing_time,
            success=True
        )

    except Exception as e:
        processing_time = int((time.time() - start_time) * 1000)
        processing_time = max(processing_time, 1)
        print(e)
        return FactCheckResponse(
            original_text=request.text,
            is_political=False,
            extracted_claims=[],
            fact_check_results=[],
            processing_time_ms=processing_time,
            success=False,
            message=f"Error processing claim: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
