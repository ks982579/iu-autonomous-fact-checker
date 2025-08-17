"""
Pulling logic from Jupyter Notebook.
Provides a class to use the latest model built for giving claim verdicts.
"""

from collections import Counter
import json
import sqlite3
import pandas as pd
from datasets import Dataset
from pathlib import Path
import logging
import re

import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    pipeline
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils import resample, shuffle
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
from datetime import datetime
from typing import List
import os
from enum import Enum

import shutil
import glob

from abc import ABC, abstractmethod
# TODO: Pull out of here and move to separate package
# NOTE: https://docs.python.org/3/library/abc.html
class UseLazyModelABC(ABC):
    @abstractmethod
    def give_verdict(self):
        ... # Implement yourself please

# from notebook
LabelMap = Enum(
    'LabelMap', 
    [
        ('SUPPORTS', 0),
        ('REFUTES', 1),
        ('NOT ENOUGH INFO', 2),
    ]
)

class ClaimJudge(UseLazyModelABC):
    """
    Running BERT fine-tuned model to judge validity of cliams based on evidence provided.
    """
    __class_name__ = "ClaimJudge"

    def __init__(self):
        self._current_dir = Path(__file__).resolve().parent
        self._model_path = self._current_dir / "trainingresults" / "latest"
        # TODO: Make Lazy Later
        self._model = None
        # TODO: Check the model exists to fail early
    
    @property
    def judge_model(self):
        # TODO: Checks that the model exists before getting to this point...
        if self._model is None:
            self._model = pipeline(
                task="text-classification",
                model=str(self._model_path),
                tokenizer=str(self._model_path),
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        return self._model

    @judge_model.setter
    def judge_model(self, val):
        raise Exception("Cannot set this value")
    
    def transform_input(self, claim: str, evidence: List[str]):
        if isinstance(evidence, list):
            evidence_text = " ".join(evidence)
        else:
            evidence_text = str(evidence)
        
        # Update for BERT Specific
        return f"[CLS] CLAIM: {claim} [SEP] EVIDENCE: {evidence_text} [SEP]"

    # THINK: could be __call__?
    def give_verdict(self, claim: str, evidence: List[str]):
        fixed_input = self.transform_input(claim, evidence)
        return self.judge_model(fixed_input)
        # RETURNS a list of results...
    
    def __call__(self, claim: str, evidence: List[str]):
        return self.give_verdict(claim, evidence)
    

if __name__ == "__main__":
    # NOTE: Specific to me for now
    val_file_path = Path(__file__).resolve().parent / ".datasets" / "processed" / "fever_dev_chunk_0000.parquet"
    assert val_file_path.exists()
    val_df = pd.read_parquet(val_file_path)

    nx = 20

    val_mini = pd.concat([
        val_df[val_df['label'] == 'SUPPORTS'].sample(n=nx, replace=False, ignore_index=True),
        val_df[val_df['label'] == 'REFUTES'].sample(n=nx, replace=False, ignore_index=True),
        val_df[val_df['label'] == 'NOT ENOUGH INFO'].sample(n=nx, replace=False, ignore_index=True),
    ])

    # Shuffle
    val_mini = val_mini.sample(frac=1, replace=False, ignore_index=True)

    print()
    print(val_mini['label'].value_counts())
    print("-"*30)

    the_judge = ClaimJudge()

    lresults = [] # [expected, predicted]

    for index, ts in val_mini.iterrows():
        results = the_judge(ts['claim'], ts['evidence'])
        lresults.append({
            'expected': ts['label'],
            'predicted': results[0]
        })

    mycounts = {}
    for res in lresults:
        pred = res.get("predicted")
        p_label = pred.get('label')
        if "0" in p_label:
            p_label = "SUPPORTS"
        elif "1" in p_label:
            p_label = "REFUTES"
        elif "2" in p_label:
            p_label = "NOT ENOUGH INFO"
        else:
            p_label = "ERROR - WHAT?!?"
        exp = res.get("expected")

        try:
            mycounts[p_label][exp] += 1
        except:
            one = mycounts.get(p_label)
            if one is None:
                mycounts[p_label] = {}
            mycounts[p_label][exp] = 1

    print(json.dumps(mycounts, indent=2))
    print()
