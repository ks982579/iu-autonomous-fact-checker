#!/usr/bin/env python3
"""
Simple rule-based political content classifier as a backup
"""
import re
from typing import Dict, List

class SimplePoliticalClassifier:
    """
    A simple rule-based political content classifier that uses keyword matching
    and patterns to identify political content. This serves as a fallback when
    the trained model is not performing well.
    """
    
    def __init__(self):
        # Political keywords and phrases
        self.political_keywords = {
            # Government and institutions
            'government', 'congress', 'senate', 'house', 'parliament', 'legislature',
            'administration', 'cabinet', 'supreme court', 'federal', 'state',
            
            # Political figures and parties
            'trump', 'biden', 'harris', 'obama', 'clinton', 'pelosi', 'mcconnell',
            'republican', 'democrat', 'democratic', 'gop', 'liberal', 'conservative',
            'libertarian', 'green party', 'progressive',
            
            # Elections and voting
            'election', 'vote', 'voting', 'ballot', 'campaign', 'candidate',
            'primary', 'caucus', 'polling', 'voter', 'electoral',
            
            # Policy and issues
            'policy', 'bill', 'law', 'legislation', 'amendment', 'constitution',
            'healthcare', 'immigration', 'tax', 'taxes', 'economy', 'budget',
            'deficit', 'debt ceiling', 'climate change', 'abortion', 'gun control',
            
            # Political processes
            'politics', 'political', 'partisan', 'bipartisan', 'filibuster',
            'impeachment', 'scandal', 'corruption', 'lobby', 'lobbying',
            
            # International politics
            'nato', 'un', 'united nations', 'diplomacy', 'sanctions', 'treaty',
            'putin', 'china', 'russia', 'ukraine', 'brexit', 'eu', 'european union',
            
            # Media and political terms
            'poll', 'polling', 'approval rating', 'debate', 'town hall',
            'press conference', 'white house', 'capitol', 'washington dc'
        }
        
        # Political hashtags and social media patterns
        self.political_hashtags = {
            'maga', 'kag', 'trump2024', 'biden2024', 'politics', 'election2024',
            'vote', 'voting', 'democrat', 'republican', 'gop', 'liberal',
            'conservative', 'progressive', 'resist', 'walkaway'
        }
        
        # Non-political indicators (strong indicators of non-political content)
        self.non_political_indicators = {
            # Entertainment
            'movie', 'film', 'tv show', 'netflix', 'spotify', 'music', 'song',
            'concert', 'album', 'artist', 'celebrity', 'hollywood',
            
            # Food and lifestyle
            'food', 'recipe', 'cooking', 'restaurant', 'coffee', 'pizza',
            'breakfast', 'lunch', 'dinner', 'delicious', 'yummy',
            
            # Sports
            'football', 'basketball', 'baseball', 'soccer', 'tennis', 'golf',
            'olympics', 'nfl', 'nba', 'mlb', 'game', 'match', 'score',
            
            # Personal life
            'family', 'friends', 'pet', 'dog', 'cat', 'vacation', 'travel',
            'birthday', 'anniversary', 'wedding', 'baby', 'kids', 'children',
            
            # Technology (non-political)
            'app', 'game', 'software', 'iphone', 'android', 'computer',
            'internet', 'social media', 'instagram', 'tiktok', 'youtube',
            
            # Nature and weather
            'weather', 'sunny', 'rain', 'snow', 'beautiful', 'sunset',
            'beach', 'park', 'nature', 'flowers', 'garden'
        }
    
    def calculate_political_score(self, text: str) -> float:
        """
        Calculate a political score based on keyword matching
        
        Args:
            text (str): Text to analyze
            
        Returns:
            float: Score between 0 and 1 (higher = more political)
        """
        text_lower = text.lower()
        
        # Count political keywords
        political_count = 0
        for keyword in self.political_keywords:
            if keyword in text_lower:
                # Weight longer keywords more heavily
                weight = max(1, len(keyword.split()))
                political_count += weight
        
        # Check for political hashtags
        for hashtag in self.political_hashtags:
            if f'#{hashtag}' in text_lower or hashtag in text_lower:
                political_count += 2  # Hashtags are strong indicators
        
        # Count non-political indicators (negative score)
        non_political_count = 0
        for indicator in self.non_political_indicators:
            if indicator in text_lower:
                non_political_count += 1
        
        # Calculate score
        total_words = len(text.split())
        political_density = political_count / max(total_words, 1)
        non_political_density = non_political_count / max(total_words, 1)
        
        # Base score on political density, penalize for non-political content
        score = political_density * 5 - non_political_density * 2
        
        # Normalize to 0-1 range
        score = max(0, min(1, score))
        
        return score
    
    def classify_content(self, text: str, confidence_threshold: float = 0.3) -> Dict:
        """
        Classify text as political or non-political
        
        Args:
            text (str): Text to classify
            confidence_threshold (float): Threshold for classification
            
        Returns:
            dict: Classification result
        """
        if not text or len(text.strip()) < 5:
            return {
                'text': text,
                'label': 'insufficient_content',
                'confidence': 0.0,
                'is_political': False,
                'method': 'simple_rule_based'
            }
        
        score = self.calculate_political_score(text)
        is_political = score >= confidence_threshold
        
        # Confidence is the score if political, 1-score if non-political
        confidence = score if is_political else (1 - score)
        
        return {
            'text': text,
            'label': 'political' if is_political else 'non_political',
            'confidence': confidence,
            'is_political': is_political,
            'political_score': score,
            'method': 'simple_rule_based'
        }
    
    def classify_social_media_post(self, post: str, confidence_threshold: float = 0.3) -> Dict:
        """
        Classify a social media post with sentence-level analysis
        
        Args:
            post (str): Social media post text
            confidence_threshold (float): Threshold for classification
            
        Returns:
            dict: Classification result with breakdown
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+\s+', post)
        sentence_results = []
        political_count = 0
        total_score = 0
        
        for sentence in sentences:
            if len(sentence.strip()) > 10:
                result = self.classify_content(sentence, confidence_threshold)
                sentence_results.append(result)
                
                if result['is_political']:
                    political_count += 1
                total_score += result['political_score']
        
        if not sentence_results:
            return self.classify_content(post, confidence_threshold)
        
        # Overall classification
        political_ratio = political_count / len(sentence_results)
        avg_score = total_score / len(sentence_results)
        
        is_political = political_ratio > 0.5 or avg_score > confidence_threshold
        confidence = avg_score if is_political else (1 - avg_score)
        
        return {
            'text': post,
            'overall_label': 'political' if is_political else 'non_political',
            'overall_confidence': confidence,
            'is_political': is_political,
            'political_sentence_ratio': political_ratio,
            'sentence_breakdown': sentence_results,
            'method': 'simple_rule_based'
        }

def main():
    """Test the simple classifier"""
    classifier = SimplePoliticalClassifier()
    
    test_texts = [
        "Trump announces new immigration policy changes",
        "Biden's healthcare reform bill passes Congress",
        "I love pizza and going to movies with friends",
        "My cat is so cute when she sleeps",
        "The election results are being contested",
        "Just watched a great Netflix show last night",
        "Congressional hearing reveals corruption",
        "Beautiful sunset walk in the park today"
    ]
    
    print("Simple Rule-Based Political Classifier Test:")
    print("=" * 60)
    
    for text in test_texts:
        result = classifier.classify_content(text, confidence_threshold=0.3)
        print(f"Text: {text}")
        print(f"Result: {'Political' if result['is_political'] else 'Non-Political'}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Political Score: {result['political_score']:.3f}")
        print("-" * 40)

if __name__ == "__main__":
    main()