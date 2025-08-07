from transformers import pipeline
from pathlib import Path
import torch
try:
    from .simple_political_classifier import SimplePoliticalClassifier
except ImportError:
    from simple_political_classifier import SimplePoliticalClassifier

class PoliticalContentClassifier:
    """
    A lightweight political content detection classifier
    Optimized for speed and small model size while maintaining accuracy
    """
    __class_name__ = "PoliticalContentClassifier"

    def __init__(self, model_path=None, use_simple_fallback=True):
        current_dir = Path(__file__).resolve().parent
        
        self.use_simple_fallback = use_simple_fallback
        self.simple_classifier = SimplePoliticalClassifier() if use_simple_fallback else None
        
        if model_path is None:
            model_path = current_dir / 'trainingresults' / 'latest'
        
        # Check if CUDA is available for better performance
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self._model = None
        try:
            if model_path.exists():
                self._model = pipeline(
                    task="text-classification",
                    model=str(model_path),
                    tokenizer=str(model_path),
                    device=device
                )
                print(f"Political classifier loaded on {device}")
            else:
                print(f"Model path {model_path} not found")
        except Exception as e:
            print(f"Failed to load trained model: {e}")
        
        if self._model is None and use_simple_fallback:
            print("Using simple rule-based classifier as fallback")
    
    def classify_content(self, text: str, confidence_threshold: float = 0.5):
        """
        Classify text content as political or non-political
        
        Args:
            text (str): Text content to classify
            confidence_threshold (float): Minimum confidence for classification
            
        Returns:
            dict: Classification result with label, confidence, and is_political boolean
        """
        if not text or len(text.strip()) < 5:
            return {
                'text': text,
                'label': 'insufficient_content',
                'confidence': 0.0,
                'is_political': False
            }
        
        # Use simple classifier if trained model is not available or not performing well
        if self._model is None or self.use_simple_fallback:
            return self.simple_classifier.classify_content(text, confidence_threshold)
        
        # Clean text for better classification
        clean_text = self._preprocess_text(text)
        
        # Run classification with trained model
        result = self._model(clean_text)
        
        # Parse results based on model training
        # From training data: 0 = non-political, 1 = political
        # HuggingFace pipeline returns LABEL_0 and LABEL_1
        label = result[0]['label']
        confidence = result[0]['score']
        
        # LABEL_1 corresponds to class 1 (political), LABEL_0 to class 0 (non-political)
        is_political = label == 'LABEL_1'
        
        # Fallback to simple classifier if confidence is very low or result seems wrong
        if confidence < 0.7 and self.simple_classifier is not None:
            simple_result = self.simple_classifier.classify_content(text, confidence_threshold)
            
            # Use simple classifier result if it has higher confidence
            if simple_result['confidence'] > confidence:
                return simple_result
        
        # Apply confidence threshold
        if confidence < confidence_threshold:
            is_political = False
            label = 'uncertain'
        else:
            label = 'political' if is_political else 'non_political'
        
        return {
            'text': text,
            'label': label,
            'confidence': confidence,
            'is_political': is_political,
            'method': 'trained_model'
        }
    
    def classify_social_media_post(self, post: str, confidence_threshold: float = 0.6):
        """
        Special method for social media posts that may contain multiple sentences
        
        Args:
            post (str): Social media post content
            confidence_threshold (float): Higher threshold for social media
            
        Returns:
            dict: Overall classification with sentence-level breakdown
        """
        # Split post into sentences for detailed analysis
        sentences = self._split_sentences(post)
        sentence_results = []
        political_count = 0
        total_confidence = 0
        
        for sentence in sentences:
            if len(sentence.strip()) > 10:  # Filter out short fragments
                result = self.classify_content(sentence, confidence_threshold)
                sentence_results.append(result)
                
                if result['is_political']:
                    political_count += 1
                total_confidence += result['confidence']
        
        if not sentence_results:
            return self.classify_content(post, confidence_threshold)
        
        # Determine overall classification
        political_ratio = political_count / len(sentence_results)
        avg_confidence = total_confidence / len(sentence_results)
        
        # Post is political if >50% of sentences are political or any sentence is highly political
        is_political = (
            political_ratio > 0.5 or 
            any(r['confidence'] > 0.8 and r['is_political'] for r in sentence_results)
        )
        
        return {
            'text': post,
            'overall_label': 'political' if is_political else 'non_political',
            'overall_confidence': avg_confidence,
            'is_political': is_political,
            'political_sentence_ratio': political_ratio,
            'sentence_breakdown': sentence_results
        }
    
    def _preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess text for better classification
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Cleaned text
        """
        import re
        
        # Remove URLs
        text = re.sub(r'http\S+', ' ', text)
        
        # Remove mentions and hashtags for cleaner analysis (but keep the text)
        text = re.sub(r'@\w+', ' ', text)
        text = re.sub(r'#(\w+)', r'\1', text)  # Keep hashtag content
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Limit length for model constraints (BERT-like models have 512 token limit)
        if len(text) > 400:  # Conservative limit leaving room for tokenization
            text = text[:400] + "..."
        
        return text
    
    def _split_sentences(self, text: str) -> list:
        """
        Split text into sentences with improved handling of social media content
        
        Args:
            text (str): Text to split
            
        Returns:
            list: List of sentences
        """
        import re
        
        # Handle common social media sentence patterns
        # Split on periods, exclamations, questions, but be careful with URLs and abbreviations
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Clean up sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 5:
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def batch_classify(self, texts: list, confidence_threshold: float = 0.6):
        """
        Classify multiple texts efficiently
        
        Args:
            texts (list): List of texts to classify
            confidence_threshold (float): Confidence threshold
            
        Returns:
            list: List of classification results
        """
        results = []
        for text in texts:
            result = self.classify_content(text, confidence_threshold)
            results.append(result)
        
        return results

def load_political_classifier(model_path=None):
    """
    Convenience function to load the political content classifier
    
    Args:
        model_path (str, optional): Path to trained model
        
    Returns:
        PoliticalContentClassifier: Loaded classifier
    """
    return PoliticalContentClassifier(model_path)

# Example usage and testing
def main():
    classifier = PoliticalContentClassifier()
    
    # Test examples
    test_texts = [
        "I love pizza and going to the movies",
        "The new tax policy will impact small businesses significantly",
        "Trump's latest speech about immigration reform",
        "Just watched a great movie on Netflix tonight!",
        "The congressional hearing revealed important facts about healthcare",
        "My cat is so cute when she sleeps"
    ]
    
    print("Testing Political Content Classifier:")
    print("=" * 50)
    
    for text in test_texts:
        result = classifier.classify_content(text)
        print(f"Text: {text}")
        print(f"Result: {result['label']} (confidence: {result['confidence']:.3f})")
        print(f"Is Political: {result['is_political']}")
        print("-" * 30)

if __name__ == "__main__":
    main()