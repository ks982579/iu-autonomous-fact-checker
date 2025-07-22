# main.py
"""
This is the entrypoint for fact-checking software.
Current implementation is in Python for more rapid development.

WARNING: current implementation of RAG w/ChromaDB depends on running file from this root directory.
L> This must change, but for now, run it from here!
"""
import json
from pathlib import Path
import requests as Requests
from typing import List, Dict, Any
# Home Grown
from aieng import claim_normalizer
from aieng.claim_normalizer.normalizer import extract_search_keywords
from aieng.rag_system.rag_eng import get_news, NewsApiJsonResponse, SimpleScraper

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
## Spelling and Grammer Check
## Normalization of text
## Extract keyword


# Rag System


def main():
    fake_statements = fake_extraction()
    
    keyword_list: List[str] = []
    for statement in fake_statements:
        keyword_list.append(extract_search_keywords(statement, 10))
    
    that = "string space"
    
    # Get News URLs
    # newses: List[Requests.Response] = []
    for look in keyword_list:
        # Get list of articles (only 25 for now)
        # sep=None for any whitespace
        news = get_news(look.split(sep=None))

        # TODO: Join Articles with the NewsApiJsonResponse Class I created
        # article metadata
        for article_md in news.json().get("articles"):
            metadata = NewsApiJsonResponse(article_md)
            # logging
            logpath = None
            if True:
                logpath = Path(f'./logging/{''.join(metadata.source.casefold().split(sep=None))}/{''.join(metadata.title[:12].casefold().split(sep=None))}')
                logpath.mkdir(exist_ok=True, parents=True)
                with open(logpath / "file.log", 'a') as file:
                    ...
                    ## json loads or dumps... then continue

            scraper = SimpleScraper()
            scraper.scrape_article_content(metadata.url)



if __name__ == "__main__":
    print("Welcome!")
else:
    raise Exception("This file is not meant for importing.")