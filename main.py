# main.py
"""
This is the entrypoint for fact-checking software.
Current implementation is in Python for more rapid development.

WARNING: current implementation of RAG w/ChromaDB depends on running file from this root directory.
L> This must change, but for now, run it from here!
"""
from datetime import datetime
import json
from pathlib import Path
import requests as Requests
from typing import List, Dict, Any
import re
import threading
# Home Grown
## TODO: Future - One 'models' package with properly named files to hold classes?
from aieng.claim_extractor import ClaimExtractor
from aieng import claim_normalizer
from aieng.claim_normalizer.normalizer import extract_search_keywords
from aieng.rag_system.rag_eng import get_news, NewsApiJsonResponseArticle, SimpleScraper, ScrapeArticleResponse
from aieng.rag_system.vectordb import VectorPipeline

# Constants
LOGGING = True
class LoggingJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            # Resort to '__dict__'
            if hasattr(obj, "__dict__"):
                result = {}
                for k, v in obj.__dict__.items():
                    try:
                        # test first if serializable
                        json.dumps(v)
                        result[k] = v
                    except TypeError:
                        # else make into string
                        result[k] = str(v)
                return result
            else:
                return str(obj)

def stringify(obj, indent=2):
    return json.dumps(obj, indent=indent, cls=LoggingJsonEncoder)

class MyLogger:
    def __init__(self, active: bool, logpath = None, logfile = None):
        self.active = active
        if not self.active:
            return

        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        this_file = Path(__file__).resolve()

        # should be getters and setters
        if logpath is None:
            self.logpath = Path(f'{this_file.parent}/logging/{self.timestamp}') ## puts logger in same directory as main file
        elif logpath is Path:
            self.logpath = logpath
        else:
            raise TypeError("Parameter 'logpath' must be type Path")

        if logfile is None:
            self.logfile = Path("main.log") ## puts logger in same directory as main file
        elif logfile is Path:
            self.logfile = logfile
        else:
            raise TypeError("Parameter 'logfile' must be type Path")
            
        self.logpath.mkdir(exist_ok=True, parents=True)
        ## Setting up Log File
        self.write_log({
            "action": "creating this log file",
            "timestamp": datetime.now().isoformat()
        })

    def write_log(self, log_obj: object):
        if not self.active:
            return
        with open(self.logpath / self.logfile, 'a', encoding='utf-8') as file:
            try:
                content = stringify(log_obj, indent=2)
                file.write(content+"\n")
                print(content)
            except Exception as err:
                print(f"Logging Error: {err}")
    
    # parameters are tightly coupled
    def log_article(self, metadata, scraper_response):
        if not self.active:
            return
        try:
            # Logging article content too - different file
            articlepath = self.logpath / Path(f"{''.join(metadata.source.site_name.casefold().split(sep=None))}/{''.join(metadata.title[:12].casefold().split(sep=None))}")
            articlepath = self.logpath / Path(f"{self._only_alphanum(metadata.source.site_name)}/{self._only_alphanum(metadata.title)[:12]}")
            articlepath.mkdir(exist_ok=True, parents=True)

            ## Setting up Log File
            with open(articlepath / "article.log", 'w', encoding='utf-8') as file:
                first_log = {
                    "action": "creating this log file",
                    "timestamp": datetime.now().isoformat()
                }
                file.write(stringify(first_log)+"\n")
                if scraper_response.content is not None:
                    file.write(scraper_response.content)
        except Exception as err:
            print(f"Logging Error: {err}")
    
    #region Private Methods
    @staticmethod
    def _only_alphanum(content: str) -> str:
        """
        For logging articles with proper directory names.
        Some article titles have punctuation that should not be in directory name.

        Args:
            content (str): string to be made only lowercase alphanumeric

        Returns:
            str: lowercase alphanumeric content - also no whitespace
        """
        expression = r'[^A-Za-z0-9]+'
        return re.sub(expression, '', content.casefold())

# -----------------------------------
class MyThreadSafeLogger:
    def __init__(self, active: bool, logpath = None, logfile = None):
        self.active = active
        if not self.active:
            return
        
        # TODO: private...
        self.logger = MyLogger(active, logpath, logfile) # This will create file
        self.lock = threading.Lock()
    
    def write_log(self, log_obj: object):
        if not self.active:
            return
        with self.lock:
            self.logger.write_log(log_obj)
    
    # parameters are tightly coupled
    def log_article(self, metadata, scraper_response):
        """
        This should be writing to a new location so shouldn't need lock.

        Args:
            metadata (_type_): _description_
            scraper_response (_type_): _description_
        """
        if not self.active:
            return
        self.logger.log_article(metadata, scraper_response)

# Twitter Tweets

fake_tweet = ""

# Claim Extraction

def fake_extraction() -> List[str]: 
    return [
        "The president announced a new climate policy yesterday.",
        "Apple's stock price increased by 15% last week.", 
        # "COVID-19 vaccines are 95% effective against severe illness.",
        # "Tesla stock price dropped 20% this week.",
        # "Microsoft announced a new AI partnership yesterday.",
        # "Biden signed a climate change bill last month.",
        # "Apple released a new iPhone model in September.",
        "Meta laid off 10,000 employees in 2024.",
        "Trump is going to deport Elon Musk!",
    ]

def fake_post_request() -> List[str]:
    return [
        "My opponent Denver Riggleman, running mate of Corey Stewart, was caught on camera campaigning with a white supremacist. Now he has been exposed as a devotee of Bigfoot erotica. This is not what we need on Capitol Hill."
    ]


# Claim Normalization
## Spelling and Grammer Check
## Normalization of text
## Extract keyword

"""
Idea - Check similarity of individual claims
  No point in checking the same claim twice because it's worded differently. 
"""

"""
Idea - To start project, if ChromaDB is empty maybe reach out to NewsAPI.org for current events.
"""

"""
Idea - Part of Normalizing Claim
 * the query string passed to NewsAPI.org is a mess.
 * this is where 5w1h comes into play
"""

# Rag System

# /logging/<time-stamp>/

"""
Idea could be to create a new - smaller - vector store for similarity search.
Only store articles that are used for judgment? 
Could keep vector store lean - but more API requests and scraping perhaps. 
"""

def main():
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    this_file = Path(__file__).resolve()
    logpath = Path(f'{this_file.parent}/logging/{timestamp}') ## puts logger in same directory as main file
    logfile = Path("main.log") ## overall log filename 
    ## Setup only if logging

    logger = MyThreadSafeLogger(LOGGING) # This will create file

    # fake_statements: List[str] = fake_extraction()
    fake_posts: List[str] = fake_post_request() # s/b fake_posts

    # Do we want the model to be loaded as needed, or loaded here in the background?
    claim_model = ClaimExtractor()

    ## HERE - Begin Multi Threading
    ### TODO: 
    ## ----------------------------

    ## TODO: claim_scores: typing
    claim_scores = []

    ## TODO: First check if Post is Political...
    for post in fake_posts:
        claim_scores = claim_model.assign_claim_score_to_text(post)
        print(claim_scores)
        logger.write_log({
            "action": "assign claim score to text",
            "timestamp": datetime.now().isoformat(),
            "claim_scores": claim_scores,
        })

    # Because multiple statements, there are multiple lists of keywords
    keyword_list: List[List[str]] = []

    fake_statements = []
    ## TODO: Function for this part I think
    for score in claim_scores:
        if score['label'] == 'Claim': # Not checking confidence currently
            fake_statements.append(score['text'])

    ## for each statement, extract the keywords
    # TODO: Better extraction process - removing certain words currently
    for statement in fake_statements:
        keyword_list.append(extract_search_keywords(statement, 10))

    # Create the VectorPipeline
    vector_pipeline = VectorPipeline()
    
    # Get News URLs
    # NOTE: each elm in loop is an extracted claim... in end, we actually might use all?
    for look in keyword_list:
        # Get list of articles (only 25 for now)
        # sep=None to split by any whitespace
        ## TODO: Probably need to ensure removal of characters like emojis
        news_api_response = get_news(look, page_size=10) # 10 for testing
        news_api_json = news_api_response.json()

        logger.write_log({
            "action": "maked request to newsapi.org",
            "timestamp": datetime.now().isoformat(),
            "html_status_code": news_api_response.status_code,
            "news_api_response": news_api_response.__dict__,
            "news_api_json": news_api_json,
        })

        # TODO: Join Articles with the NewsApiJsonResponse Class I created
        # article metadata
        for article_md in news_api_response.json().get("articles"): # {
            metadata = NewsApiJsonResponseArticle(article_md)

            logger.write_log({
                "action": "put article data into python class",
                "timestamp": datetime.now().isoformat(),
                "article": metadata
            })

            return ## Testing up to here
            
            # Before Scraping - Check if it already exists in vector store
            already_stored = vector_pipeline.article_exists(metadata.url)
            # Consider: if article is revised or updated but with same url?

            logger.write_log({
                "action": "check if article already stored in vectore db",
                "timestamp": datetime.now().isoformat(),
                "article_url": metadata.url,
                "article_title": metadata.title,
                "article_already_in_vector_store": already_stored
            })

            # skip the next part of scraping if we already have it in store.
            if already_stored:
                continue

            # TODO: Improve scraper to get only the article data.
            scraper = SimpleScraper(timeout=10, delay=1) # These are default but I am explicit
            # Poorly scraping article
            scraper_response: ScrapeArticleResponse = scraper.scrape_article_content(metadata)

            logger.write_log({
                "action": "get and scrape article content",
                "timestamp": datetime.now().isoformat(),
                "scraper_response": {
                    "url": scraper_response.url,
                    "success": scraper_response.success,
                    "error": scraper_response.error,
                }
            })
            logger.log_article(metadata, scraper_response)


            # Now update the Vector Store - Only if we could scrape
            if scraper_response.content is None:
                continue

            vector_result = vector_pipeline.add_article({
                'url': metadata.url,
                'title': metadata.title,
                'source': metadata.source.site_name,
                'author': metadata.author,
                'published_at': metadata.published_at,
                'full_content': scraper_response.content
            }) # Can produce exception

            logger.write_log({
                "action": "added article to vector db",
                "timestamp": datetime.now().isoformat(),
                "add_article_result": vector_result
            })

        # } end looping through articles
        
        
    # Now we search vector store for results
    # TODO: statements probably need to be cleaned for best results
    for statement in fake_statements:
        keyword_list.append(extract_search_keywords(statement, 10))
        search_similar_results = vector_pipeline.search_similar_content(
            query=statement,
            n_results=10,
        )

        logger.write_log({
            "action": "searched for similar content",
            "timestamp": datetime.now().isoformat(),
            "similarity_search_results": search_similar_results,
        })
        
        if not search_similar_results.get("success", False):
            print("Not sure how to handle if no results... should be because we checked API...")
            # skip for now
            continue
            
        print("For this claim, we pass information to the judge AI or back to the backend.")

    logger.write_log({
        "action": "all done",
        "timestamp": datetime.now().isoformat(),
    })


if __name__ == "__main__":
    print("Welcome!")
    print(Path(__file__).resolve().parent)
    main()
else:
    raise Exception("This file is not meant for importing.")