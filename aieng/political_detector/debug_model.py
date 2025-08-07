#!/usr/bin/env python3
from transformers import pipeline
from pathlib import Path
import torch

def debug_model():
    # Load the model directly
    current_dir = Path(__file__).resolve().parent
    model_path = current_dir / 'trainingresults' / 'latest'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = pipeline(
        task="text-classification",
        model=str(model_path),
        tokenizer=str(model_path),
        device=device,
        return_all_scores=True  # Get scores for all classes
    )
    
    # Test clearly political and non-political texts
    test_texts = [
        "Trump announces new immigration policy",
        "Biden's healthcare reform bill passes",
        "I love pizza and going to movies",
        "My cat is sleeping on the couch",
        "Congressional hearing reveals corruption",
        "Walking in the park on a sunny day"
    ]
    
    print("Raw model outputs:")
    print("=" * 50)
    
    for text in test_texts:
        result = model(text)
        print(f"Text: {text}")
        print(f"Raw result: {result}")
        
        # Show which label has higher score
        if len(result[0]) == 2:
            label_0_score = result[0][0]['score']  # LABEL_0
            label_1_score = result[0][1]['score']  # LABEL_1
            predicted_label = result[0][0]['label'] if label_0_score > label_1_score else result[0][1]['label']
            max_score = max(label_0_score, label_1_score)
            print(f"Predicted: {predicted_label} (score: {max_score:.4f})")
            
            # According to training: 0=non-political, 1=political
            if predicted_label == 'LABEL_1':
                print("Interpretation: POLITICAL")
            else:
                print("Interpretation: NON-POLITICAL")
        
        print("-" * 30)

if __name__ == "__main__":
    debug_model()