#!/usr/bin/env python3
from political_classifier import PoliticalContentClassifier

def quick_test():
    classifier = PoliticalContentClassifier()
    
    # Test a clearly political text
    political_text = "Trump announces new immigration policy changes"
    result = classifier.classify_content(political_text, confidence_threshold=0.5)
    
    print(f"Text: {political_text}")
    print(f"Result: {result}")
    print()
    
    # Test a clearly non-political text
    non_political_text = "I love pizza and movies"
    result2 = classifier.classify_content(non_political_text, confidence_threshold=0.5)
    
    print(f"Text: {non_political_text}")
    print(f"Result: {result2}")

if __name__ == "__main__":
    quick_test()