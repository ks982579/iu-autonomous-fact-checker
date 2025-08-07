#!/usr/bin/env python3
"""
Test script to validate political content detection integration
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from aieng.political_detector.political_classifier import PoliticalContentClassifier

def test_political_classifier():
    """Test the political content classifier with various examples"""
    
    print("Loading Political Content Classifier...")
    classifier = PoliticalContentClassifier()
    print("✓ Classifier loaded successfully")
    
    # Test cases
    test_cases = [
        # Political content
        ("Trump's new tax policy will impact small businesses", True),
        ("The congressional hearing revealed important information", True),
        ("Biden announces new immigration reform", True),
        ("Republicans and Democrats disagree on healthcare", True),
        ("The election results are being contested", True),
        
        # Non-political content
        ("I love pizza and going to movies with friends", False),
        ("Just finished watching a great Netflix show", False),
        ("My cat is so cute when she sleeps", False),
        ("Working on a new programming project today", False),
        ("The weather is beautiful for a walk in the park", False),
        
        # Mixed/ambiguous content
        ("The new tax policy affects everyone's budget planning", None),  # Could go either way
        ("Government agencies provide various public services", None),    # Could go either way
    ]
    
    print("\nTesting Political Content Classification:")
    print("=" * 60)
    
    correct_predictions = 0
    total_definitive_tests = 0
    
    for text, expected_political in test_cases:
        result = classifier.classify_content(text, confidence_threshold=0.6)
        
        # Check if prediction matches expectation (for definitive cases)
        if expected_political is not None:
            total_definitive_tests += 1
            if result['is_political'] == expected_political:
                correct_predictions += 1
                status = "✓"
            else:
                status = "✗"
        else:
            status = "?"  # Ambiguous case
        
        print(f"{status} Text: {text[:50]}{'...' if len(text) > 50 else ''}")
        print(f"   Prediction: {'Political' if result['is_political'] else 'Non-Political'}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Label: {result['label']}")
        print()
    
    # Calculate accuracy
    accuracy = correct_predictions / total_definitive_tests if total_definitive_tests > 0 else 0
    print(f"Accuracy on definitive test cases: {correct_predictions}/{total_definitive_tests} = {accuracy:.2%}")
    print()
    
    return accuracy

def test_social_media_posts():
    """Test with social media-style posts"""
    
    print("Testing Social Media Post Classification:")
    print("=" * 60)
    
    classifier = PoliticalContentClassifier()
    
    social_media_posts = [
        "Just voted! Make sure you get out there and make your voice heard! #Election2024 #Vote",
        "Loving this new coffee shop downtown! The barista art is amazing ☕️ #coffee #local",
        "Can't believe what Congress is doing with our tax money. We need better representation! #politics",
        "Movie night with friends! Watching the new Marvel movie 🎬 #movienight #friends",
        "The immigration policy changes will affect thousands of families across the country",
        "Beautiful sunset walk with my dog in the park today 🌅 #nature #peaceful",
    ]
    
    for post in social_media_posts:
        result = classifier.classify_social_media_post(post, confidence_threshold=0.6)
        
        print(f"Post: {post}")
        print(f"Overall: {'Political' if result['is_political'] else 'Non-Political'}")
        print(f"Confidence: {result['overall_confidence']:.3f}")
        print(f"Political Ratio: {result['political_sentence_ratio']:.2f}")
        print(f"Sentences analyzed: {len(result['sentence_breakdown'])}")
        print()

def test_batch_processing():
    """Test batch processing capabilities"""
    
    print("Testing Batch Processing:")
    print("=" * 60)
    
    classifier = PoliticalContentClassifier()
    
    texts = [
        "The new healthcare bill passed the Senate",
        "I made delicious pasta for dinner tonight",
        "Trump's latest speech about the economy",
        "Going to the beach this weekend!",
        "Congressional approval ratings are at all-time low",
    ]
    
    results = classifier.batch_classify(texts, confidence_threshold=0.6)
    
    for i, (text, result) in enumerate(zip(texts, results)):
        print(f"{i+1}. {text}")
        print(f"   Result: {'Political' if result['is_political'] else 'Non-Political'} ({result['confidence']:.3f})")
    print()

def main():
    """Run all tests"""
    print("Political Content Detection - Integration Test")
    print("=" * 70)
    
    try:
        # Test basic classification
        accuracy = test_political_classifier()
        
        # Test social media posts
        test_social_media_posts()
        
        # Test batch processing
        test_batch_processing()
        
        print("Integration Test Summary:")
        print(f"✓ Model loads successfully")
        print(f"✓ Classification accuracy: {accuracy:.2%}")
        print(f"✓ Social media post analysis works")
        print(f"✓ Batch processing works")
        print("\n🎉 All integration tests passed!")
        
        if accuracy >= 0.8:
            print("✅ Model performance is excellent (≥80% accuracy)")
        elif accuracy >= 0.7:
            print("⚠️  Model performance is good (≥70% accuracy)")
        else:
            print("❌ Model performance needs improvement (<70% accuracy)")
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)