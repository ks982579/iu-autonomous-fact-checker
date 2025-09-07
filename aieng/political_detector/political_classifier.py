"""
Copied from the notebook so I can load the model and use it. 
"""
from pathlib import Path
import torch
import re
from enum import Enum
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    pipeline
)

# From Training File
class FileConfig:
    __FullRunContext = """denotes using the complete dataset of just a smaller portion of it for testing."""
    FullRun = True

    __MiniRunContext = """denotes using a mini dataset for testing. If FullRun is True this is ignored."""
    MiniRun = False

    __ModelVersionContext = """To keep things in order, you may set model version here."""
    ModelVersion = "v0.1.0"

    ## NOTE: currently not in use for this file
    __Percentage = """If NOT a full run, what percentage of data do we use? (Think decimal values 0 < pc < 1)"""
    Percentage = 0.1
    
    __ToBuildContext = """Do we want to build another model or run without build for testing purposes."""
    ToBuild = True

    __CustomLossFnContext = """For certain cases with class imbalance we need a custom loss function."""
    CustomLossFn = False
    
    __ChunkOverlapContext = """For the vector store, chunks are 64 words with 8 word overlap."""
    ChunkOverlap = 8

    # Don't know why an option, it's basically required.
    UsePadding = True

    # BaseModelName = "bert-base-uncased" # Context too small
    # BaseModelName = "answerdotai/ModernBERT-base"
    BaseModelName = "distilbert-base-uncased" # all lowercase
    MaxTokens = 512 # Max for DistilBERT
    Hardware = 'cuda' if torch.cuda.is_available() else 'cpu'

# class Labels(Enum)
LabelMap = Enum(
    'LabelMap', 
    [
        ('political', 0),
        ('other', 1),
        # ('NOT ENOUGH INFO', 2),
    ]
)
# print(f"HARDWARE: {FileConfig.Hardware}")

class PoliticalTextClassifier:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Ensure token length
        self.tokenizer.model_max_length = FileConfig.MaxTokens
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=FileConfig.Hardware,
        )
        self.model.eval()
    
    def __call__(self, text):
        # To not track gradients to save memory - don't need back propagation
        with torch.no_grad():
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True,
                # max_length=FileConfig.MaxTokens, 
                padding=True
            )
            inputs = {k: v.to(FileConfig.Hardware) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            
            # Format like pipeline output
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_id = torch.argmax(probs, dim=-1).item()
            
            # Same as Hugging Face
            return [{
                'label': self.model.config.id2label[pred_id],
                'score': probs[0][pred_id].item()
            }]

class PoliticalContentClassifier:
    """
    A lightweight political content detection classifier
    Optimized for speed and small model size while maintaining accuracy
    """
    __class_name__ = "PoliticalContentClassifier"

    def __init__(self, model_path=None, use_simple_fallback=True):
        current_dir = Path(__file__).resolve().parent
        
        if model_path is None:
            model_path = current_dir / 'trainingresults' / 'latest'
        
        # Check if CUDA is available for better performance
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self._model = None
        try:
            if model_path.exists():
                self._model = PoliticalTextClassifier(model_path)
                print(f"Political classifier loaded on {device}")
            else:
                print(f"Model path {model_path} not found")
        except Exception as e:
            print(f"Failed to load trained model: {e}")
        
        if self._model is None and use_simple_fallback:
            print("Using simple rule-based classifier as fallback")
    
    def classify_content(self, text: str):
        """
        Classify text content as political or non-political.
        
        Args:
            text (str): Text content to classify
            
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
        
        # Clean text for better classification
        clean_text = self._preprocess_text(text)
        print("Political Classifier:")
        print(clean_text) 
        print()
        
        # Run classification with trained model
        result_l = self._model(clean_text) 
        
        # Parse results based on model training
        # From training data: 0 = non-political, 1 = political
        # HuggingFace pipeline returns LABEL_0 and LABEL_1
        result = result_l[0] # only one at a time for now...

        prediction = result['label']
        confidence = result['score']
        print(f"{prediction} :: {confidence}")
        # 0 means political content
        is_political = "0" in prediction
        
        label = 'political' if is_political else 'non_political'
        
        return {
            'text': text,
            'label': label,
            'confidence': confidence,
            'is_political': is_political,
            'method': 'trained_model'
        }
    
    # WARN: NOT USED
    def __classify_social_media_post(self, post: str, confidence_threshold: float = 0.6):
        """
        Special method for social media posts that may contain multiple sentences
        
        Args:
            post (str): Social media post content
            confidence_threshold (float): Higher threshold for social media
            
        Returns:
            dict: Overall classification with sentence-level breakdown
        """
        ## NOTE: should read entire post now
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
        # Replace URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '{{URL}}', text)
        
        # Replace all @mentions with {{USERNAME}}
        text = re.sub(r'@\w+', '{{USERNAME}}', text)
        text = re.sub(r'\n+', ' ', text)

        # How this DistilBERT model was trained
        return f"[CLS] {text} [SEP]"
    
    # Should be able to read entire post
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
