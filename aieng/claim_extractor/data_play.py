"""
This file gathers the data I need from Sentiment 140 for claim detection model.
"""
from datetime import datetime
import logging
import os
from pathlib import Path
import re

import pandas as pd

logging.basicConfig(level=logging.INFO)

class DataLoader:
    logger = logging.getLogger(__name__)
    def __init__(self, file_path: Path):
        assert file_path.exists()

        self.file_path = file_path
        if file_path.suffix == ".parquet":
            try:
                self.dataframe = pd.read_parquet(file_path)
            except Exception as ex:
                self.logger.exception(
                    msg={
                        "message": "Error reading Parquet file",
                        "exception": ex,
                    },
                    stack_info=True,
                )
        else:
            self.logger.warning(
                msg=f"File type {file_path.suffix} is not yet supported."
            )

    
    @staticmethod
    def standardize_preprocessing(text):
        """
        Standard preprocessing for both TweetTopic and your political data
        """
        # Replace URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '{{URL}}', text)
        
        # Replace all @mentions with {{USERNAME}}
        text = re.sub(r'@\w+', '{{USERNAME}}', text)
        
        return text

    def preprocess_column(self, col_from: str, col_to: str):
        """
        Breaking columns into bins based on text length

        Args:
            col_from (str): _description_
            col_to (str): _description_
        """
        self.dataframe[f"{col_from}__length"] = self.dataframe[col_from].apply(lambda x: len(x))
        df = self.dataframe
        assert df is not None
        assert col_from in df.columns
        if col_to in df.columns:
            self.logger.warning(msg="This action will overwrite a column")
        self.dataframe[col_to] = df[col_from].apply(self.standardize_preprocessing)
        # Good to have if length is mostly URL
        self.dataframe[f'{col_to}__length'] = self.dataframe[col_to].apply(lambda x: len(x))
        df[f'{col_from}__length_bin'] = pd.qcut(
            df[f"{col_from}__length"],
            q=9,
            labels=False,
            duplicates='drop'
        )
        print(df[f"{col_from}__length_bin"].value_counts())

    
    def get_info(self):
        """
        displaying info I need to split data properly
        """
        df = self.dataframe
        assert df is not None
        print()
        print('Info:')
        print(df.info())
        print()
        print('Description:')
        print(df.describe())
        print()
        print('columns:')
        print(df.columns)
        print()
        print('Head:')
        print(df.head(5))
        print()
    
    def save_sample_csv(self, count: int, save_path: Path):
        df = self.dataframe
        assert count < len(df)

        size = count // 5

        df1 = df[df['text__length_bin'].apply(lambda x: x in [0, 1])]
        df2 = df[df['text__length_bin'].apply(lambda x: x in [2, 3])]
        df3 = df[df['text__length_bin'].apply(lambda x: x in [4, 5])]
        df4 = df[df['text__length_bin'].apply(lambda x: x in [6, 7])]
        df5 = df[df['text__length_bin'].apply(lambda x: x == 8)]

        sizedf1 = size
        sizedf5 = size

        if int(size * 1.25) < len(df5):
            sizedf5 = int(size * 1.25)
            sizedf1 = count - (sizedf5 + (size * 3))
        elif int(size * 1.15) < len(df5):
            sizedf5 = int(size * 1.15)
            sizedf1 = count - (sizedf5 + (size * 3))
        elif int(size * 1.05) < len(df5):
            sizedf5 = int(size * 1.05)
            sizedf1 = count - (sizedf5 + (size * 3))
        else:
            sizedf5 = count - (size * 4)

        newdf = pd.concat([
            df1.sample(n=sizedf1, replace=False, ignore_index=True),
            df2.sample(n=size, replace=False, ignore_index=True),
            df3.sample(n=size, replace=False, ignore_index=True),
            df4.sample(n=size, replace=False, ignore_index=True),
            df5.sample(n=sizedf5, replace=False, ignore_index=True),
        ])
        save_path.mkdir(exist_ok=True)
        datenow = datetime.now()
        timestamp = datenow.strftime("%Y%m%d%H%M%S")
        newdf.to_csv(save_path / f"sample_sentiment_140_{timestamp}.csv")
        



if __name__ == "__main__":
    print("Starting Application")
    data_dir = Path(__file__).resolve().parent / '.data_sets'
    # sent140 = DataLoader(data_dir / 'sentiment140.parquet')
    # sent140.preprocess_column("text", "cleantext")
    # sent140.get_info()
    # sent140.save_sample_csv(10000, data_dir)
    lesa = pd.read_parquet(data_dir / 'lesa' / 'lesa_set.parquet')
    lesa['label'] = lesa['claim'].apply(lambda x: 'claim' if x == 1 else 'other')
    lesa['text'] = lesa['tweet_text']
    lesa.drop(columns=['claim', 'tweet_text'],inplace=True)
    print(lesa.head())
    print()
    sent140 = pd.read_csv(data_dir / 'sample_sentiment_140_10000.csv', encoding='utf-8', index_col=0)
    sent140['label'] = 'other'
    sent140 = sent140[['text', 'label']]
    print(sent140.head())
    lesa2 = pd.concat([lesa, sent140])
    lesa2 = lesa2.sample(frac=1, replace=False, ignore_index=True)
    print(lesa2.head(10))
    print(lesa2['label'].value_counts())
    lesa2.to_parquet(data_dir / 'lesa' / 'lesa_sent140_set.parquet')
