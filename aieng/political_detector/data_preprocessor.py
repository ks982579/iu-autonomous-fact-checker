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

class PoliticalDataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for political content detection
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
        self.processed_data_path = self.datasets_dir / 'processed_data.csv'
    
    def download_sentiment140(self):
        """
        Download the Sentiment140 dataset for non-political tweets
        """
        logger.info("Downloading Sentiment140 dataset...")
        
        # Sentiment140 dataset URL
        url = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                # Extract the training data file
                zip_file.extract("training.1600000.processed.noemoticon.csv", self.datasets_dir)
                
                # Rename to our expected filename
                extracted_file = self.datasets_dir / "training.1600000.processed.noemoticon.csv"
                extracted_file.rename(self.sentiment140_path)
                
            logger.info(f"Sentiment140 dataset downloaded to {self.sentiment140_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download Sentiment140: {e}")
            return False
    
    def load_political_tweets(self) -> pd.DataFrame:
        """
        Load the political tweets dataset
        
        Returns:
            pd.DataFrame: Political tweets with label=1
        """
        if not self.political_tweets_path.exists():
            raise FileNotFoundError(f"Political tweets not found at {self.political_tweets_path}")
        
        logger.info("Loading political tweets...")
        df = pd.read_csv(self.political_tweets_path)
        
        # Use the 'text' column and assign political label
        political_df = df[['text']].copy()
        political_df['label'] = 1  # Political = 1
        political_df['source'] = 'political_tweets'
        
        logger.info(f"Loaded {len(political_df)} political tweets")
        return political_df
    
    def load_sentiment140_tweets(self, sample_size=None) -> pd.DataFrame:
        """
        Load Sentiment140 dataset as non-political tweets
        
        Args:
            sample_size (int, optional): Number of tweets to sample
            
        Returns:
            pd.DataFrame: Non-political tweets with label=0
        """
        if not self.sentiment140_path.exists():
            if not self.download_sentiment140():
                raise FileNotFoundError("Could not download or find Sentiment140 dataset")
        
        logger.info("Loading Sentiment140 tweets...")
        
        # Sentiment140 columns: polarity, id, date, query, user, text
        column_names = ['polarity', 'id', 'date', 'query', 'user', 'text']
        df = pd.read_csv(self.sentiment140_path, encoding='latin1', header=None, names=column_names)
        
        # Extract text and assign non-political label
        sentiment_df = df[['text']].copy()
        sentiment_df['label'] = 0  # Non-political = 0
        sentiment_df['source'] = 'sentiment140'
        
        # Sample if requested
        if sample_size and len(sentiment_df) > sample_size:
            sentiment_df = sentiment_df.sample(n=sample_size, random_state=42)
            logger.info(f"Sampled {sample_size} non-political tweets")
        else:
            logger.info(f"Loaded {len(sentiment_df)} non-political tweets")
        
        return sentiment_df
    
    def clean_text(self, text: str) -> str:
        """
        Clean tweet text for training
        
        Args:
            text (str): Raw tweet text
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', ' [URL] ', text)
        
        # Replace mentions with placeholder but keep some context
        text = re.sub(r'@\w+', ' [MENTION] ', text)
        
        # Clean hashtags - keep the content but remove #
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove excessive punctuation
        text = re.sub(r'([!?.]){2,}', r'\1', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove very short texts (likely noise)
        if len(text) < 10:
            return ""
        
        return text
    
    def filter_non_political_content(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out potentially political content from non-political dataset
        
        Args:
            df (pd.DataFrame): Non-political tweets dataframe
            
        Returns:
            pd.DataFrame: Filtered dataframe
        """
        logger.info("Filtering out potentially political content...")
        
        # Political keywords to exclude
        political_keywords = [
            # General politics
            'trump', 'biden', 'harris', 'president', 'congress', 'senate', 'house',
            'republican', 'democrat', 'gop', 'liberal', 'conservative', 'election',
            'vote', 'voting', 'ballot', 'campaign', 'politics', 'political',
            'government', 'policy', 'tax', 'healthcare', 'immigration',
            
            # International politics
            'putin', 'china', 'russia', 'ukraine', 'nato', 'eu', 'brexit',
            
            # Social issues that often become political
            'abortion', 'gun control', 'climate change', 'covid', 'vaccine',
            'border', 'refugee', 'supreme court',
            
            # News sources
            'cnn', 'fox news', 'msnbc', 'politico', 'reuters',
        ]
        
        # Create regex pattern
        pattern = '|'.join([rf'\b{keyword}\b' for keyword in political_keywords])
        
        # Filter out tweets containing political keywords
        initial_count = len(df)
        df_filtered = df[~df['text'].str.contains(pattern, case=False, na=False)]
        filtered_count = len(df_filtered)
        
        logger.info(f"Filtered out {initial_count - filtered_count} potentially political tweets")
        logger.info(f"Remaining non-political tweets: {filtered_count}")
        
        return df_filtered
    
    def balance_dataset(self, political_df: pd.DataFrame, non_political_df: pd.DataFrame) -> pd.DataFrame:
        """
        Balance the dataset to have equal numbers of political and non-political tweets
        
        Args:
            political_df (pd.DataFrame): Political tweets
            non_political_df (pd.DataFrame): Non-political tweets
            
        Returns:
            pd.DataFrame: Balanced dataset
        """
        political_count = len(political_df)
        non_political_count = len(non_political_df)
        
        logger.info(f"Political tweets: {political_count}")
        logger.info(f"Non-political tweets: {non_political_count}")
        
        # Use the smaller count as target
        target_count = min(political_count, non_political_count)
        
        # Sample equal amounts
        if len(political_df) > target_count:
            political_balanced = political_df.sample(n=target_count, random_state=42)
        else:
            political_balanced = political_df.copy()
        
        if len(non_political_df) > target_count:
            non_political_balanced = non_political_df.sample(n=target_count, random_state=42)
        else:
            non_political_balanced = non_political_df.copy()
        
        # Combine datasets
        balanced_df = pd.concat([political_balanced, non_political_balanced], ignore_index=True)
        
        # Shuffle the dataset
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Created balanced dataset with {len(balanced_df)} tweets")
        logger.info(f"Label distribution:")
        logger.info(balanced_df['label'].value_counts())
        
        return balanced_df
    
    def create_train_val_test_split(self, df: pd.DataFrame, 
                                  test_size: float = 0.2, 
                                  val_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train, validation, and test sets
        
        Args:
            df (pd.DataFrame): Full dataset
            test_size (float): Proportion for test set
            val_size (float): Proportion for validation set (from remaining data)
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: train, val, test dataframes
        """
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
        """
        Save processed datasets to files
        
        Args:
            train_df (pd.DataFrame): Training set
            val_df (pd.DataFrame): Validation set
            test_df (pd.DataFrame): Test set
        """
        # Save individual splits
        train_path = self.datasets_dir / 'train.csv'
        val_path = self.datasets_dir / 'val.csv'
        test_path = self.datasets_dir / 'test.csv'
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        # Save combined dataset
        combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        combined_df.to_csv(self.processed_data_path, index=False)
        
        logger.info(f"Saved datasets to {self.datasets_dir}")
        logger.info(f"Train: {train_path}")
        logger.info(f"Val: {val_path}")
        logger.info(f"Test: {test_path}")
        logger.info(f"Combined: {self.processed_data_path}")
    
    def preprocess_full_pipeline(self, max_non_political: int = 700000):
        """
        Run the complete preprocessing pipeline
        
        Args:
            max_non_political (int): Maximum number of non-political tweets to use
        """
        logger.info("Starting full preprocessing pipeline...")
        
        try:
            # Load political tweets
            political_df = self.load_political_tweets()
            
            # Load non-political tweets
            non_political_df = self.load_sentiment140_tweets(sample_size=max_non_political)
            
            # Clean text data
            logger.info("Cleaning text data...")
            political_df['text'] = political_df['text'].apply(self.clean_text)
            non_political_df['text'] = non_political_df['text'].apply(self.clean_text)
            
            # Remove empty texts after cleaning
            political_df = political_df[political_df['text'].str.len() > 0]
            non_political_df = non_political_df[non_political_df['text'].str.len() > 0]
            
            # Filter out political content from non-political dataset
            non_political_df = self.filter_non_political_content(non_political_df)
            
            # Balance the dataset
            balanced_df = self.balance_dataset(political_df, non_political_df)
            
            # Create train/val/test splits
            train_df, val_df, test_df = self.create_train_val_test_split(balanced_df)
            
            # Save processed datasets
            self.save_processed_datasets(train_df, val_df, test_df)
            
            # Print final statistics
            logger.info("Preprocessing completed successfully!")
            logger.info(f"Final dataset size: {len(balanced_df)} tweets")
            logger.info(f"Political: {len(balanced_df[balanced_df['label'] == 1])}")
            logger.info(f"Non-political: {len(balanced_df[balanced_df['label'] == 0])}")
            
            return train_df, val_df, test_df
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise

def main():
    """
    Run the preprocessing pipeline
    """
    preprocessor = PoliticalDataPreprocessor()
    train_df, val_df, test_df = preprocessor.preprocess_full_pipeline()
    
    print("Preprocessing completed!")
    print(f"Train set: {len(train_df)} tweets")
    print(f"Validation set: {len(val_df)} tweets") 
    print(f"Test set: {len(test_df)} tweets")

if __name__ == "__main__":
    main()