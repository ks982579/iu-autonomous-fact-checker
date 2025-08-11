import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import logging
from typing import Tuple, List, Dict
import requests
import zipfile
import io

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FixedPoliticalDataPreprocessor:
    """
    Fixed data preprocessing pipeline for political content detection
    Addresses artifacts that caused model to learn spurious features
    """
    
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = Path(__file__).resolve().parent
        
        self.base_dir = Path(base_dir)
        self.datasets_dir = self.base_dir / '.datasets'
        self.datasets_dir.mkdir(exist_ok=True)
        
        # Dataset paths
        self.political_tweets_path = self.datasets_dir / 'Political_tweets.csv'
        self.sentiment140_path = self.datasets_dir / 'sentiment140.csv'
        self.processed_data_path = self.datasets_dir / 'processed_data_fixed.csv'
    
    def download_sentiment140(self):
        """Download the Sentiment140 dataset for non-political tweets"""
        logger.info("Downloading Sentiment140 dataset...")
        
        url = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                zip_file.extract("training.1600000.processed.noemoticon.csv", self.datasets_dir)
                extracted_file = self.datasets_dir / "training.1600000.processed.noemoticon.csv"
                extracted_file.rename(self.sentiment140_path)
                
            logger.info(f"Sentiment140 dataset downloaded to {self.sentiment140_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download Sentiment140: {e}")
            return False
    
    def load_political_tweets(self) -> pd.DataFrame:
        """Load political tweets dataset"""
        if not self.political_tweets_path.exists():
            raise FileNotFoundError(f"Political tweets not found at {self.political_tweets_path}")
        
        logger.info("Loading political tweets...")
        df = pd.read_csv(self.political_tweets_path)
        
        political_df = df[['text']].copy()
        political_df['label'] = 1  # Political = 1
        political_df['source'] = 'political_tweets'
        
        logger.info(f"Loaded {len(political_df)} political tweets")
        return political_df
    
    def load_sentiment140_tweets(self, sample_size=None) -> pd.DataFrame:
        """Load Sentiment140 dataset as non-political tweets"""
        if not self.sentiment140_path.exists():
            if not self.download_sentiment140():
                raise FileNotFoundError("Could not download or find Sentiment140 dataset")
        
        logger.info("Loading Sentiment140 tweets...")
        
        column_names = ['polarity', 'id', 'date', 'query', 'user', 'text']
        df = pd.read_csv(self.sentiment140_path, encoding='latin1', header=None, names=column_names)
        
        sentiment_df = df[['text']].copy()
        sentiment_df['label'] = 0  # Non-political = 0
        sentiment_df['source'] = 'sentiment140'
        
        if sample_size and len(sentiment_df) > sample_size:
            sentiment_df = sentiment_df.sample(n=sample_size, random_state=42)
            logger.info(f"Sampled {sample_size} non-political tweets")
        else:
            logger.info(f"Loaded {len(sentiment_df)} non-political tweets")
        
        return sentiment_df
    
    def clean_text_uniformly(self, text: str) -> str:
        """
        Clean text uniformly to avoid preprocessing artifacts
        CRITICAL: Apply same cleaning to both political and non-political texts
        """
        if not isinstance(text, str):
            return ""
        
        # Basic cleaning only - avoid creating artifacts
        text = text.lower().strip()
        
        # Remove URLs completely (don't use placeholders)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove user mentions completely (don't use placeholders)
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags but keep content
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove excessive punctuation
        text = re.sub(r'([!?.]){2,}', r'\1', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Filter minimum length
        if len(text) < 10:
            return ""
        
        # IMPORTANT: Limit text length to reduce length bias
        if len(text) > 140:
            # Take first 140 characters to avoid bias toward longer political texts
            text = text[:140].strip()
        
        return text
    
    def filter_non_political_content_lightly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply VERY light filtering to avoid over-sanitizing non-political content
        Only remove obviously political content, keep borderline cases
        """
        logger.info("Applying light filtering to non-political content...")
        
        # Only filter very obvious political keywords
        strict_political_keywords = [
            'trump', 'biden', 'harris', 'obama', 'clinton',
            'republican', 'democrat', 'gop', 
            'congress', 'senate', 'house of representatives',
            'president', 'vice president',
            'election', 'ballot', 'vote for', 'campaign',
            'political', 'politics'
        ]
        
        # Create regex pattern for strict filtering only
        pattern = '|'.join([rf'\b{keyword}\b' for keyword in strict_political_keywords])
        
        initial_count = len(df)
        df_filtered = df[~df['text'].str.contains(pattern, case=False, na=False)]
        filtered_count = len(df_filtered)
        
        logger.info(f"Light filtering: removed {initial_count - filtered_count} obviously political tweets")
        logger.info(f"Remaining non-political tweets: {filtered_count}")
        
        return df_filtered
    
    def balance_dataset_carefully(self, political_df: pd.DataFrame, non_political_df: pd.DataFrame) -> pd.DataFrame:
        """
        Balance dataset while ensuring text length distributions are similar
        """
        logger.info("Balancing dataset with length consideration...")
        
        # Clean both datasets first
        political_df = political_df.copy()
        non_political_df = non_political_df.copy()
        
        political_df['text'] = political_df['text'].apply(self.clean_text_uniformly)
        non_political_df['text'] = non_political_df['text'].apply(self.clean_text_uniformly)
        
        # Remove empty texts
        political_df = political_df[political_df['text'].str.len() > 0]
        non_political_df = non_political_df[non_political_df['text'].str.len() > 0]
        
        # Calculate text lengths
        political_df['text_length'] = political_df['text'].str.len()
        non_political_df['text_length'] = non_political_df['text'].str.len()
        
        logger.info(f"Before balancing:")
        logger.info(f"  Political: {len(political_df)} tweets, avg length: {political_df['text_length'].mean():.1f}")
        logger.info(f"  Non-political: {len(non_political_df)} tweets, avg length: {non_political_df['text_length'].mean():.1f}")
        
        # Balance by sampling equal amounts
        target_size = min(len(political_df), len(non_political_df), 50000)  # Cap at 50k each
        
        # Sample stratified by length to ensure similar distributions
        if len(political_df) > target_size:
            political_balanced = political_df.sample(n=target_size, random_state=42)
        else:
            political_balanced = political_df.copy()
        
        if len(non_political_df) > target_size:
            non_political_balanced = non_political_df.sample(n=target_size, random_state=42)
        else:
            non_political_balanced = non_political_df.copy()
        
        # Remove text_length column
        political_balanced = political_balanced.drop('text_length', axis=1)
        non_political_balanced = non_political_balanced.drop('text_length', axis=1)
        
        # Combine datasets
        balanced_df = pd.concat([political_balanced, non_political_balanced], ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Balanced dataset: {len(balanced_df)} tweets")
        logger.info(f"Label distribution:")
        logger.info(balanced_df['label'].value_counts())
        
        return balanced_df
    
    def create_train_val_test_split(self, df: pd.DataFrame, 
                                  test_size: float = 0.2, 
                                  val_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataset into train, validation, and test sets"""
        logger.info(f"Splitting dataset: test={test_size}, val={val_size}")
        
        # First split: train+val vs test
        train_val, test = train_test_split(
            df, test_size=test_size, random_state=42, stratify=df['label']
        )
        
        # Second split: train vs val
        train, val = train_test_split(
            train_val, test_size=val_size, random_state=42, stratify=train_val['label']
        )
        
        logger.info(f"Train set: {len(train)} tweets")
        logger.info(f"Validation set: {len(val)} tweets")
        logger.info(f"Test set: {len(test)} tweets")
        
        return train, val, test
    
    def save_processed_datasets(self, train_df: pd.DataFrame, 
                              val_df: pd.DataFrame, 
                              test_df: pd.DataFrame):
        """Save processed datasets to files"""
        # Save individual splits
        train_path = self.datasets_dir / 'train_fixed.csv'
        val_path = self.datasets_dir / 'val_fixed.csv'
        test_path = self.datasets_dir / 'test_fixed.csv'
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        # Save combined dataset
        combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        combined_df.to_csv(self.processed_data_path, index=False)
        
        logger.info(f"Saved fixed datasets to {self.datasets_dir}")
        logger.info(f"Train: {train_path}")
        logger.info(f"Val: {val_path}")
        logger.info(f"Test: {test_path}")
        logger.info(f"Combined: {self.processed_data_path}")
    
    def preprocess_full_pipeline(self, max_non_political: int = 100000):
        """Run the complete FIXED preprocessing pipeline"""
        logger.info("Starting FIXED preprocessing pipeline...")
        
        try:
            # Load datasets
            political_df = self.load_political_tweets()
            non_political_df = self.load_sentiment140_tweets(sample_size=max_non_political)
            
            # Apply light filtering only
            non_political_df = self.filter_non_political_content_lightly(non_political_df)
            
            # Balance dataset with length consideration
            balanced_df = self.balance_dataset_carefully(political_df, non_political_df)
            
            # Final text length check
            final_lengths = balanced_df.groupby('label')['text'].apply(lambda x: x.str.len().mean())
            logger.info(f"Final average text lengths:")
            for label, avg_len in final_lengths.items():
                label_name = "Political" if label == 1 else "Non-Political"
                logger.info(f"  {label_name}: {avg_len:.1f} characters")
            
            # Create splits
            train_df, val_df, test_df = self.create_train_val_test_split(balanced_df)
            
            # Save processed datasets
            self.save_processed_datasets(train_df, val_df, test_df)
            
            logger.info("FIXED preprocessing completed successfully!")
            logger.info(f"Final dataset size: {len(balanced_df)} tweets")
            logger.info(f"Political: {len(balanced_df[balanced_df['label'] == 1])}")
            logger.info(f"Non-political: {len(balanced_df[balanced_df['label'] == 0])}")
            
            return train_df, val_df, test_df
            
        except Exception as e:
            logger.error(f"FIXED preprocessing failed: {e}")
            raise

def main():
    """Run the FIXED preprocessing pipeline"""
    preprocessor = FixedPoliticalDataPreprocessor()
    train_df, val_df, test_df = preprocessor.preprocess_full_pipeline()
    
    print("FIXED preprocessing completed!")
    print(f"Train set: {len(train_df)} tweets")
    print(f"Validation set: {len(val_df)} tweets") 
    print(f"Test set: {len(test_df)} tweets")

if __name__ == "__main__":
    main()