"""
Pulling logic from Jupyter Notebook.
Provides a class to use the latest model built for giving claim verdicts.
Update: 2025-08-17
Previously did not use ModernBERT nor F16 and did not enforce the token window length.
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
from typing import List, Optional
import os
from enum import Enum

import shutil
import glob

from abc import ABC, abstractmethod

# This is more like a private class for this file
class FileConfig:
    __ChunkOverlapContext = """For the vector store, chunks are 64 words with 8 word overlap."""
    ChunkOverlap = 8

    # Don't know why an option, it's basically required.
    UsePadding = True

    # BaseModelName = "bert-base-uncased" # Context too small
    BaseModelName = "answerdotai/ModernBERT-base"
    # I think ModernBert allows for 512 * 16 = 8192
    MaxTokens = 512 * 2 # Current latest build
    Hardware = 'cuda' if torch.cuda.is_available() else 'cpu'

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

class ClaimJudgeTextClassifier:
    def __init__(self, model_path: Optional[Path] = None):
        if model_path is None:
            model_path = Path(__file__).resolve().parent / 'trainingresults' / 'latest'
        assert model_path.exists()
        model_str_path = str(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_str_path)
        # Ensure token length
        self.tokenizer.model_max_length = FileConfig.MaxTokens
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_str_path,
            torch_dtype=torch.float16,
            device_map=FileConfig.Hardware,
        )
        self.model.eval()
    
    @staticmethod
    def _translate_input(claim: str, evidence: List[str]):
        if isinstance(evidence, list):
            evidence_text = " ".join(evidence)
        else:
            evidence_text = str(evidence)
        
        # Update for BERT Specific
        return f"[CLS] CLAIM: {claim} [SEP] EVIDENCE: {evidence_text} [SEP]"
    
    def __call__(self, claim: str, evidence: List[str]):
        # To not track gradients to save memory - don't need back propagation
        text = self._translate_input(claim, evidence)
        with torch.no_grad():
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True,
                padding=True
            )
            inputs = {k: v.to(FileConfig.Hardware) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            
            # Format like HF Transformers pipeline output
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_id = torch.argmax(probs, dim=-1).item()
            
            # Same as Hugging Face
            return [{
                'label': self.model.config.id2label[pred_id],
                'score': probs[0][pred_id].item()
            }]

# Note: I think the following is getting deprecated
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
