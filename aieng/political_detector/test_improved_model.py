#!/usr/bin/env python3
"""
Test script for the improved political content detection model
Verifies that the model works correctly and doesn't have the previous issues
"""

from transformers import pipeline
from pathlib import Path
import torch

def test_improved_model():
    """Test the improved political content detection model"""
    
    # Look for the latest improved model
    current_dir = Path(__file__).resolve().parent
    
    # Try different possible model paths
    model_paths = [
        current_dir / 'trainingresults' / 'latest_basic_improved',
        current_dir / 'trainingresults' / 'latest_simple_improved', 
        current_dir / 'trainingresults' / 'latest_improved'
    ]
    
    model_path = None
    for path in model_paths:
        if path.exists():
            model_path = path
            break
    
    if model_path is None:
        # Fall back to any improved model directory
        results_dir = current_dir / 'trainingresults'
        improved_dirs = [d for d in results_dir.glob('*improved*') if d.is_dir()]
        if improved_dirs:
            model_path = improved_dirs[-1]  # Get most recent
    
    if model_path is None:
        print("ERROR: No improved model found!")
        print("Available models:")
        results_dir = current_dir / 'trainingresults'
        for path in results_dir.iterdir():
            if path.is_dir():
                print(f"  {path.name}")
        return
    
    print(f"Testing model: {model_path}")
    
    # Load the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        model = pipeline(
            task="text-classification",
            model=str(model_path),
            tokenizer=str(model_path),
            device=device,
            return_all_scores=True
        )
        print(f"Model loaded successfully on {device}")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return
    
    # Test with clearly political and non-political texts
    # These should be correctly classified without relying on artifacts
    test_cases = [
        # Political texts (should be classified as political)
        ("Trump announces new immigration policy", "Political"),
        ("Biden's healthcare reform bill passes", "Political"), 
        ("Congressional hearing reveals corruption", "Political"),
        ("New senator elected in swing state", "Political"),
        ("Supreme court decision affects voting rights", "Political"),
        
        # Non-political texts (should be classified as non-political)  
        ("I love pizza and going to movies", "Non-Political"),
        ("My cat is sleeping on the couch", "Non-Political"),
        ("Walking in the park on a sunny day", "Non-Political"),
        ("Just watched a great movie on Netflix", "Non-Political"),
        ("The weather is beautiful today", "Non-Political"),
        
        # Edge cases (harder to classify)
        ("I disagree with this decision", "Ambiguous"),
        ("This is important for our future", "Ambiguous"),
        ("People should have the right to choose", "Ambiguous")
    ]
    
    print("\nTesting improved model:")
    print("=" * 80)
    
    correct_predictions = 0
    total_clear_cases = 0  # Only count clear political/non-political cases
    
    for text, expected_category in test_cases:
        try:
            result = model(text)
            
            # Get the prediction
            label_0_score = result[0][0]['score']  # LABEL_0 
            label_1_score = result[0][1]['score']  # LABEL_1
            
            predicted_label = result[0][0]['label'] if label_0_score > label_1_score else result[0][1]['label']
            max_score = max(label_0_score, label_1_score)
            
            # Interpret the prediction (assuming 0=non-political, 1=political)
            if predicted_label == 'LABEL_1':
                prediction = "Political"
            else:
                prediction = "Non-Political"
            
            # Check if prediction is correct (only for clear cases)
            if expected_category != "Ambiguous":
                total_clear_cases += 1
                if prediction == expected_category:
                    correct_predictions += 1
                    status = "✅ CORRECT"
                else:
                    status = "❌ WRONG"
            else:
                status = "⚪ AMBIGUOUS"
            
            print(f"Text: {text}")
            print(f"Expected: {expected_category}, Predicted: {prediction} (confidence: {max_score:.3f}) {status}")
            print(f"Raw scores: LABEL_0={label_0_score:.3f}, LABEL_1={label_1_score:.3f}")
            print("-" * 60)
            
        except Exception as e:
            print(f"ERROR processing '{text}': {e}")
    
    # Calculate accuracy for clear cases
    if total_clear_cases > 0:
        accuracy = (correct_predictions / total_clear_cases) * 100
        print(f"\nACCURACY ON CLEAR CASES: {accuracy:.1f}% ({correct_predictions}/{total_clear_cases})")
        
        if accuracy >= 80:
            print("✅ MODEL PERFORMANCE: GOOD")
        elif accuracy >= 60:
            print("⚠️ MODEL PERFORMANCE: MODERATE") 
        else:
            print("❌ MODEL PERFORMANCE: POOR")
    
    print(f"\nModel path: {model_path}")

def main():
    """Main test function"""
    test_improved_model()

if __name__ == "__main__":
    main()