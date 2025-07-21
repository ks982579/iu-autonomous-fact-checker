import sqlite3
import os
from pathlib import Path
import requests #https://requests.readthedocs.io/en/latest/
from typing import Dict
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin, urlparse
import re

# News-API.org has a python client library but we will use Requests for now
# https://newsapi.org/sources - sources...
# 

"""
NewsAPI.Org
This is going to be where we start 
"""
def read_dot_env():
    dotenv_filepath: Path = Path(__file__).resolve().parent / ".env"
    assert dotenv_filepath.exists() # File should exist 
    with open(dotenv_filepath, 'r') as file:
        for line in file:
            if len(line.strip()) > 0:
                keyval_pair = [x.strip() for x in line.split("=")]
                print(keyval_pair)
                assert len(keyval_pair) == 2
                os.environ.setdefault(
                    key=keyval_pair[0],
                    value=keyval_pair[1]
                )

def get_news():
    apikey = os.getenv("NEWSAPI_APIKEY")

    headers = {
        "X-Api-Key": apikey
    }

    response = requests.request(
        "GET",
        "https://newsapi.org/v2/everything",
        params={
            "q": "trump,\"big beautiful bill\"",
            "searchIn": "content",
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": 25,
            "page": 1
        },
        headers=headers
    )

    print(response)

    return response

class ArticleSource:
    def __init__(self, data: Dict[str, None | str]):
        self.id = data.get('id')
        self.site_name = data.get('name')

class NewsApiJsonResponse:
    def __init__(self, rawdata):
        self.data = rawdata
        self.source = ArticleSource(rawdata.get('source'))
        self.author = rawdata.get('author')
        self.title = rawdata.get('title')
        self.description = rawdata.get('description')
        self.url = rawdata.get('url')
        self.url_to_image = rawdata.get('urlToImage')
        self.published_at = rawdata.get('publishedAt')
        self.content = rawdata.get('content')
    
    def __str__(self):
        result = ''
        result += "Source: \n"
        result += f"  id: {self.source.id}" + "\n"
        result += f"  name: {self.source.site_name}" + "\n"
        result += f"Author: {self.author}" + "\n"
        result += f"Title: {self.title}" + "\n"
        result += f"Title: {self.title}" + "\n"
        result += f"Description: {self.description}" + "\n"
        result += f"URL: {self.url}" + "\n"
        result += f"Published: {self.published_at}" + "\n"
        result += f"Content: {self.content}" + "\n"
        return result

# -- Review

class SimpleScraper:
    def __init__(self, timeout=10, delay=1):
        self.timeout = timeout
        self.delay = delay
        self.session = requests.Session()
        # Add headers to appear more like a real browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Elements to remove - you can add more later
        self.unwanted_tags = [
            'script', 
            'style', 
            'img', 
            'figure', 
            'figcaption', 
            'nav', 
            'footer', # 'header', 'aside', 'form'
        ]
    
    def scrape_article_content(self, url):
        """
        Simple scraper that gets body content and removes unwanted elements
        """
        try:
            # Add delay to be respectful to servers
            time.sleep(self.delay)
            
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Get the body element
            body = soup.body
            if not body:
                body = soup  # Fallback if no body tag
            
            # Remove unwanted elements
            for tag_name in self.unwanted_tags:
                for tag in body.find_all(tag_name):
                    tag.decompose()  # Remove from tree completely
            
            # Extract just the text content
            content = body.get_text()
            
            # Basic cleanup - remove extra whitespace
            content = ' '.join(content.split())
            
            return {
                'url': url,
                'content': content,
                'word_count': len(content.split()),
                'success': True
            }
            
        except Exception as e:
            return {
                'url': url,
                'content': None,
                'error': str(e),
                'success': False
            }
    
    def add_unwanted_tag(self, tag_name):
        """Add another tag type to remove"""
        if tag_name not in self.unwanted_tags:
            self.unwanted_tags.append(tag_name)

# First Usage Function
def scrape_newsapi_articles(newsapi_response):
    """
    Process NewsAPI response and scrape full content for each article
    """
    scraper = SimpleScraper()
    enriched_articles = []
    
    for article in newsapi_response.get('articles', []):
        url = article.get('url')
        if not url:
            continue
            
        print(f"Scraping: {url}")
        
        # Scrape full content
        scraped_content = scraper.scrape_article_content(url)
        
        # Combine NewsAPI metadata with scraped content
        enriched_article = {
            'source': article.get('source', {}).get('name'),
            'author': article.get('author'),
            'title': article.get('title'),
            'description': article.get('description'),
            'url': url,
            'published_at': article.get('publishedAt'),
            'full_content': scraped_content.get('content'),
            'word_count': scraped_content.get('word_count', 0),
            'scrape_success': scraped_content.get('success', False)
        }
        
        enriched_articles.append(enriched_article)
    
    return enriched_articles

# ---------



def happy_test():
    print("happy test")

def connect_sqlite3():
    connection = sqlite3.connect(":memory:") # returns Connection obj
    cursor = connection.cursor() # returns Cursor obja

if __name__ == "__main__":
    read_dot_env()
    happy_test()