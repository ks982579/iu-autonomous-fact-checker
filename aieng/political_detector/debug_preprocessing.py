#!/usr/bin/env python3
import pandas as pd

def analyze_training_data():
    """Analyze the training data to see if preprocessing removed too much information"""
    
    df = pd.read_csv('/home/ksull18/code/iu-autonomous-fact-checker/aieng/political_detector/.datasets/train.csv')
    
    # Sample political and non-political texts
    political_samples = df[df['label'] == 1]['text'].head(10).tolist()
    non_political_samples = df[df['label'] == 0]['text'].head(10).tolist()
    
    print("POLITICAL SAMPLES (label=1):")
    print("=" * 50)
    for i, text in enumerate(political_samples[:5], 1):
        print(f"{i}. {text[:100]}...")
    
    print("\nNON-POLITICAL SAMPLES (label=0):")
    print("=" * 50)
    for i, text in enumerate(non_political_samples[:5], 1):
        print(f"{i}. {text[:100]}...")
    
    # Check lengths
    political_lengths = df[df['label'] == 1]['text'].str.len()
    non_political_lengths = df[df['label'] == 0]['text'].str.len()
    
    print(f"\nTEXT LENGTH STATISTICS:")
    print(f"Political texts - Mean: {political_lengths.mean():.1f}, Median: {political_lengths.median():.1f}")
    print(f"Non-political texts - Mean: {non_political_lengths.mean():.1f}, Median: {non_political_lengths.median():.1f}")
    
    # Check for key political terms
    political_keywords = ['politics', 'election', 'vote', 'trump', 'biden', 'congress', 'government']
    
    print(f"\nKEYWORD PRESENCE:")
    for keyword in political_keywords:
        pol_count = df[df['label'] == 1]['text'].str.contains(keyword, case=False).sum()
        non_pol_count = df[df['label'] == 0]['text'].str.contains(keyword, case=False).sum()
        print(f"{keyword}: Political={pol_count}, Non-political={non_pol_count}")

if __name__ == "__main__":
    analyze_training_data()