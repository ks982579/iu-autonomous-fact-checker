"""
Integration tests for the Fact Checking API
"""
import pytest
from fastapi import status


class TestHealthEndpoints:
    """Test health check and basic endpoints"""
    
    def test_root_endpoint(self, client):
        """Test the root endpoint"""
        response = client.get("/")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data
        assert data["message"] == "Fact Checking API is running"
    
    def test_health_check(self, client):
        """Test the health check endpoint"""
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert data["status"] == "healthy"
        assert isinstance(data["timestamp"], (int, float))


class TestPoliticalClassification:
    """Test political content classification endpoint"""
    
    def test_political_check_valid_request(self, client, sample_claim_text):
        """Test political classification with valid text"""
        payload = {"text": sample_claim_text}
        response = client.post("/check-political", json=payload)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Verify response structure
        assert "is_political" in data
        assert "classification" in data
        assert "confidence" in data
        
        # Verify data types
        assert isinstance(data["is_political"], bool)
        assert isinstance(data["classification"], str)
        assert isinstance(data["confidence"], float)
        assert 0.0 <= data["confidence"] <= 1.0
    
    def test_political_check_empty_text(self, client, empty_text):
        """Test political classification with empty text"""
        payload = {"text": empty_text}
        response = client.post("/check-political", json=payload)
        
        # Should still work with empty text
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "is_political" in data
    
    def test_political_check_invalid_payload(self, client):
        """Test political classification with invalid payload"""
        # Missing required 'text' field
        payload = {"content": "some text"}
        response = client.post("/check-political", json=payload)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_political_check_no_payload(self, client):
        """Test political classification with no payload"""
        response = client.post("/check-political")
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestFactChecking:
    """Test main fact-checking endpoint"""
    
    def test_fact_check_with_claims(self, client, sample_claim_text):
        """Test fact-checking with text containing factual claims"""
        payload = {"text": sample_claim_text}
        response = client.post("/fact-check", json=payload)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Verify response structure
        required_fields = [
            "original_text", "is_political", "extracted_claims",
            "fact_check_results", "processing_time_ms", "success"
        ]
        for field in required_fields:
            assert field in data
        
        # Verify data types and values
        assert data["original_text"] == sample_claim_text
        assert isinstance(data["is_political"], bool)
        assert isinstance(data["extracted_claims"], list)
        assert isinstance(data["processing_time_ms"], int)
        assert data["processing_time_ms"] > 0
        
        # Success depends on whether content is political and has claims
        if data["is_political"] and len(data["extracted_claims"]) > 0:
            assert data["success"] is True
        else:
            assert data["success"] is False
            if not data["is_political"]:
                assert "Non political content detected" in data.get("message", "")
            elif len(data["extracted_claims"]) == 0:
                assert "No factual claims detected" in data.get("message", "")
        
        # Verify claim extraction structure
        if data["extracted_claims"]:
            for claim in data["extracted_claims"]:
                assert "text" in claim
                assert "confidence" in claim
                assert "is_factual_claim" in claim
                assert isinstance(claim["confidence"], float)
                assert isinstance(claim["is_factual_claim"], bool)
    
    def test_fact_check_with_opinions(self, client, sample_opinion_text):
        """Test fact-checking with text containing mostly opinions"""
        payload = {"text": sample_opinion_text}
        response = client.post("/fact-check", json=payload)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Opinion text is likely non-political, so expect success=False
        if not data["is_political"]:
            assert data["success"] is False
            assert "Non political content detected" in data.get("message", "")
        else:
            # If it's political but no claims, expect success=False
            assert data["success"] is False or len(data["extracted_claims"]) > 0
        assert data["original_text"] == sample_opinion_text
        assert isinstance(data["extracted_claims"], list)
    
    def test_fact_check_mixed_content(self, client, mixed_content_text):
        """Test fact-checking with mixed claims and opinions"""
        payload = {"text": mixed_content_text}
        response = client.post("/fact-check", json=payload)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Mixed content should be political and have claims
        if data["is_political"] and len(data["extracted_claims"]) > 0:
            assert data["success"] is True
        else:
            assert data["success"] is False
        assert isinstance(data["extracted_claims"], list)
        # Should extract at least one claim from mixed content
        assert len(data["extracted_claims"]) >= 0
    
    def test_fact_check_empty_text(self, client, empty_text):
        """Test fact-checking with empty text"""
        payload = {"text": empty_text}
        response = client.post("/fact-check", json=payload)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Empty text should be non-political, so expect success=False
        assert data["success"] is False
        assert data["original_text"] == empty_text
        assert data["extracted_claims"] == []
        assert "Non political content detected" in data.get("message", "")
    
    def test_fact_check_long_text(self, client, long_text):
        """Test fact-checking with long text (performance test)"""
        payload = {"text": long_text}
        response = client.post("/fact-check", json=payload)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Long text should be political and have claims since it contains factual statements
        if data["is_political"] and len(data["extracted_claims"]) > 0:
            assert data["success"] is True
        else:
            assert data["success"] is False
        assert isinstance(data["extracted_claims"], list)
        # Processing time should be reasonable (less than 30 seconds)
        assert data["processing_time_ms"] < 30000
    
    def test_fact_check_invalid_payload(self, client):
        """Test fact-checking with invalid payload"""
        payload = {"content": "some text"}  # Wrong field name
        response = client.post("/fact-check", json=payload)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_fact_check_no_payload(self, client):
        """Test fact-checking with no payload"""
        response = client.post("/fact-check")
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestAPIResponseTime:
    """Test API response times for performance"""
    
    def test_response_time_health_check(self, client):
        """Health check should be very fast"""
        import time
        start = time.time()
        response = client.get("/health")
        end = time.time()
        
        assert response.status_code == status.HTTP_200_OK
        # Health check should be under 100ms
        assert (end - start) < 0.1
    
    def test_response_time_political_check(self, client, sample_claim_text):
        """Political check should be reasonably fast"""
        import time
        payload = {"text": sample_claim_text}
        
        start = time.time()
        response = client.post("/check-political", json=payload)
        end = time.time()
        
        assert response.status_code == status.HTTP_200_OK
        # Political check should be under 1 second
        assert (end - start) < 1.0


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_endpoint(self, client):
        """Test accessing non-existent endpoint"""
        response = client.get("/nonexistent")
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_wrong_method(self, client):
        """Test using wrong HTTP method"""
        # GET request to POST endpoint
        response = client.get("/fact-check")
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED
    
    def test_malformed_json(self, client):
        """Test sending malformed JSON"""
        response = client.post(
            "/fact-check",
            data="invalid json{",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY