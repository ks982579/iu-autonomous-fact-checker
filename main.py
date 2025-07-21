# main.py
"""
This is the entrypoint for fact-checking software.
Current implementation is in Python for more rapid development.

WARNING: current implementation of RAG w/ChromaDB depends on running file from this root directory.
L> This must change, but for now, run it from here!
"""
from typing import List, Dict, Any

# Claim Extraction

def fake_extraction() -> List[str]: 
    return [
        "The president announced a new climate policy yesterday.",
        "Apple's stock price increased by 15% last week.", 
        "COVID-19 vaccines are 95% effective against severe illness.",
        "Tesla stock price dropped 20% this week.",
        "Microsoft announced a new AI partnership yesterday.",
        "Biden signed a climate change bill last month.",
        "Apple released a new iPhone model in September.",
        "Meta laid off 10,000 employees in 2024.",
        "Trump is going to deport Elon Musk!",
    ]


# Claim Normalization


# Rag System


# 


if __name__ == "__main__":
    print("Welcome!")
else:
    raise Exception("This file is not meant for importing.")