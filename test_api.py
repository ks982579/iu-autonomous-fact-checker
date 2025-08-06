#!/usr/bin/env python3
"""
Test script for the Fact Checking API
"""
import requests
import json
import time


def test_api_endpoints():
    """Test the API endpoints with sample data"""
    base_url = "http://localhost:8000"
    
    # Test data
    test_text = "The president announced a new climate policy yesterday. This is just my opinion about politics."
    
    print("🚀 Testing Fact Checking API")
    print("=" * 50)
    
    # Test 1: Health Check
    print("\n1. Testing health check endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print(f"✅ Health check passed: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ API server is not running. Start it with: python run_api.py")
        return
    
    # Test 2: Political Content Check
    print("\n2. Testing political content classification...")
    try:
        payload = {"text": test_text}
        response = requests.post(f"{base_url}/check-political", json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Political check result: {json.dumps(result, indent=2)}")
        else:
            print(f"❌ Political check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Full Fact Check
    print("\n3. Testing full fact-checking endpoint...")
    try:
        payload = {"text": test_text}
        response = requests.post(f"{base_url}/fact-check", json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Fact check result:")
            print(f"   • Original text: {result['original_text']}")
            print(f"   • Is political: {result['is_political']}")
            print(f"   • Claims found: {len(result['extracted_claims'])}")
            print(f"   • Processing time: {result['processing_time_ms']}ms")
            print(f"   • Success: {result['success']}")
            
            if result['extracted_claims']:
                print("   • Extracted claims:")
                for i, claim in enumerate(result['extracted_claims'], 1):
                    print(f"     {i}. '{claim['text']}' (confidence: {claim['confidence']:.3f})")
        else:
            print(f"❌ Fact check failed: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n✅ API testing completed!")
    print(f"API Documentation: {base_url}/docs")


if __name__ == "__main__":
    test_api_endpoints()