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

class ClaimExtractor():
    __class_name__ = "ClaimExtractor"

    def __init__(self):
        current_dir = Path(__file__).resolve().parent
        model_path = current_dir / 'trainingresults' / 'latest'
        self.hardware = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        max_token_len = getattr(self.tokenizer, 'model_max_length', 512)

        # self._model = pipeline(
        #     task="text-classification",
        #     model=str(model_path),
        #     tokenizer=str(model_path),
        #     device='cuda'
        # )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Ensure token length
        self.tokenizer.model_max_length = max_token_len
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=self.hardware,
        )
        self.model.eval()
    
    def assign_claim_score_to_text(self, sentences: str):
        """
        CURRENT: WE ASSUME SENTENCES TO BE JUST ONE...
        The string parameter can be multiple sentences, like a post from social media.
        This function will separate based on sentences and determine if they are claims.
        It will return a list of results

        Args:
            text (str): _description_
        """
        # TODO: Better typing
        claims = []
        # TODO: Could probably imporve sentence parsing
        # sentence_list = sentences.split('. ')
        # ASSUMING JUST THE ONE SENTENCE FOR NOW - BATCH LATER
        for sentence in [sentences]:
            # TODO: Possible adjustments
            if len(sentence.strip()) > 10: # filter out short statements
                result = self.classify_text(sentence)
                claims.append({
                    'text': sentence,
                    # Cound be enum?
                    ## OR -> Fact% = 1- Fake% => Might not be the same thing
                    'label': 'Claim' if result[0]['label'] == 'LABEL_1' else 'Opinion',
                    'confidence': result[0]['score']
                })
        return claims
    
    def classify_text(self, text: str):
        cleantext = self._preprocess_text(text)
        print("Claim Detector")
        print(cleantext)
        print()
        with torch.no_grad():
            inputs = self.tokenizer(
                cleantext, 
                return_tensors="pt", 
                truncation=True,
                padding=True
            )
            inputs = {k: v.to(self.hardware) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            
            # Format like pipeline output
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_id = torch.argmax(probs, dim=-1).item()
            
            # Same as Hugging Face
            return [{
                'label': self.model.config.id2label[pred_id],
                'score': probs[0][pred_id].item()
            }]

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

        # Replace newlines with spaces
        text = re.sub(r'\n+', ' ', text)

        # How this DistilBERT model was trained
        return f"[CLS] {text} [SEP]"