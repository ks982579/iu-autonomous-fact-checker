"""
Pytest configuration and fixtures for API testing
These fixtures become available in PyTest testing - very convenient.
"""
import pytest
from fastapi.testclient import TestClient
from api.main import app

# NOTE: after manual testing - I think these tests should be updated
# TODO: Create a dataset with more realistic examples

@pytest.fixture
def client():
    """
    Create a test client for the FastAPI application
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_claim_text():
    """Sample text containing factual claims"""
    return "President Biden signed a new climate change bill last month. The Senate voted 51-49 to pass the healthcare reform legislation."


@pytest.fixture
def sample_opinion_text():
    """Sample text containing mostly opinions"""
    return "I think the weather is nice today. In my opinion, cats are better than dogs."


@pytest.fixture
def mixed_content_text():
    """Sample text with both claims and opinions"""
    return "COVID-19 vaccines are 95% effective according to clinical trials. I believe this is good news for everyone."


@pytest.fixture
def empty_text():
    """Empty text for edge case testing"""
    return ""


@pytest.fixture
def long_text():
    """Long text for performance testing"""
    return " ".join([
        "Meta laid off 10,000 employees in 2024.",
        "Tesla stock price dropped 20% this week.",
        "Microsoft announced a new AI partnership yesterday.",
        "Biden signed a climate change bill last month.",
        "Apple released a new iPhone model in September.",
        "This is just my opinion about these events.",
        "I think technology companies are doing well overall."
    ])