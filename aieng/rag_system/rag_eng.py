from datetime import datetime, timedelta
import os
from pathlib import Path
import requests #https://requests.readthedocs.io/en/latest/
from typing import Dict, Generic, List, TypeVar, Union, Optional, Callable, Any
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin, urlparse
import re
import random
import json

# News-API.org has a python client library but we will use Requests for now
# https://newsapi.org/sources - sources...
# 

T = TypeVar('T')
Option = Union[T, None]

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
                assert len(keyval_pair) == 2
                os.environ.setdefault(
                    key=keyval_pair[0],
                    value=keyval_pair[1]
                )

def url_builder(protocol: str, subdomains: Optional[List[str]], second_level_domain: str, top_level_domain: str, sub_directories: Optional[List[str]]):
    """
    Tool to build URLs from inputs

    Args:
        protocol (str): _description_
        subdomains (Optional[List[str]]): _description_
        second_level_domain (str): _description_
        top_level_domain (str): _description_
        sub_directories (Optional[List[str]]): _description_

    Returns:
        _type_: _description_
    """
    assert protocol == 'http' or protocol == 'https'
    assert second_level_domain is not None and type(second_level_domain) is str
    assert top_level_domain is not None and type(top_level_domain) is str
    url = f"{protocol}://"
    if subdomains is not None and len(subdomains) > 0:
        for sub in subdomains:
            assert type(sub) is str
            url = f"{url}{sub}."
    url = f"{url}{second_level_domain}.{top_level_domain}"
    if sub_directories is not None and len(sub_directories) > 0:
        for sub in sub_directories:
            assert type(sub) is str
            url = f"{url}/{sub}"
    return url

def get_news(keywords: List[str], page_size: Option[int] = 25):
    """
    You will need to call response.json() or something depending
    on what you expect

    Args:
        keywords (List[str]): _description_
        page_size (Option<int>): Can be max of 100

    Returns:
        requests.Response: _description_
    """
    # reading from file here...
    read_dot_env()
    # TODO: should be a check first
    apikey = os.getenv("NEWSAPI_APIKEY")
    qs = " AND ".join(keywords),
    print(f"Query String: ?q={qs}")

    headers = {
        "X-Api-Key": apikey
    }

    response: requests.Response = requests.request(
        "GET",
        "https://newsapi.org/v2/everything",
        params={
            "q": qs,
            "searchIn": "content",
            "language": "en",
            "sortBy": "relevancy",
            "excludeDomains": ','.join(["siliconangle.com"]),
            "pageSize": page_size,
            "page": 1
        },
        headers=headers
    )

    print(response)

    return response


def get_sources_list():
    apikey = os.getenv("NEWSAPI_APIKEY")

    headers = {
        "X-Api-Key": apikey,
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36'
    }

    response = requests.request(
        "GET",
        "https://newsapi.org/v2/sources",
        # params={
        #     "q": "trump,\"big beautiful bill\"",
        #     "searchIn": "content",
        #     "language": "en",
        #     "sortBy": "relevancy",
        #     "pageSize": 25,
        #     "page": 1
        # },
        headers=headers
    )

    print(response)

    return response

class AllSidesNews():
    def __init__(self, timeout=10, delay=1):
        self.timeout = timeout
        self.delay = delay
        self.session = requests.Session()
        # Add headers to appear more like a real browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': "*/*",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive"
        })
        self.base_url = "https://www.allsides.com"

    def search(self, keywords: List[str]):
        # query = "+".join(keywords)
        search_url = f"{self.base_url}/search"
        params = {
            "search": " ".join(keywords),
            "item_bundle": "All",
            "sort_by": "search_api_relevance",
        }
        soup = self._get_soup(search_url, params)

        view_content = soup.find_all('div', class_='view-content')

        if not view_content:
            print("No view-content div found")
            return []
        
        search_values = view_content.find_all('div', class_='search-value')

        if not search_values:
            print("No search-value divs found")
            return []

        intermediate_urls = []
        for search_value in search_values:
            # returns first anchor tag
            anchor = search_value.find('a', class_='search-result-body')
            if anchor and anchor.get('href'):
                intermediate_url =anchor['href'] if 'http' in anchor['href'] else urljoin(self.base_url, anchor['href'])
                intermediate_urls.append(intermediate_url)

        final_stories = []

        # taking top 5 because I don't want to abuse server
        for i, intermediate_url in enumerate(intermediate_urls[:5]):
            print(f"Processing {i+1}/{len(intermediate_urls)}: {intermediate_url}")
            
            try:
                # Get the intermediate page
                time.sleep(self.delay)  # Be polite
                response = self.session.get(intermediate_url)
                response.raise_for_status()
                
                # Parse the intermediate page
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find the "Read Full Story" button
                read_more_div = soup.find('div', class_='read-more-story')
                if read_more_div:
                    final_anchor = read_more_div.find('a')
                    if final_anchor and final_anchor.get('href'):
                        final_url = final_anchor['href']
                        
                        # Get title from the intermediate page
                        title_elem = soup.find('div', class_='article-name')
                        title = title_elem.get_text(strip=True) if title_elem else "No title"
                        
                        final_stories.append({
                            'title': title,
                            'allsides_url': intermediate_url,
                            'final_url': final_url,
                            'domain': urlparse(final_url).netloc
                        })
                        print(f"Found final URL: {final_url}")
                    else:
                        print(f"No anchor in read-more-story div")
                else:
                    print(f"No read-more-story div found")
                    
            except Exception as err:
                print(f"Error processing {intermediate_url}: {err}")
                continue
        
        return final_stories
        

    def _get_soup(self, url, params):
        response = self.session.get(url,params=params, timeout=self.timeout)
        try:
            response.raise_for_status()
        except Exception as err:
            print(err)
        
        return  BeautifulSoup(response.content, 'html.parser')

class ArticleSource:
    def __init__(self, data: Dict[str, None | str]):
        self.id = data.get('id')
        self.site_name = data.get('name')

# Only for one article in the list of articles that makes up the response
class NewsApiJsonResponseArticle:
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

class ScrapeArticleResponse:
    def __init__(self):
        self.url = None
        self.content = None
        self.word_count = 0
        self.error = None
        self.success = False

# Simple Scraper is for reaching out to actual articles.
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
            'footer', 
            'aside',
            'form',
            'meta',
            'header', # Sometimes header holds the title - we don't need that
        ]

    #region Private Methods
    # This will raise uncaught exception
    def _get_soup(self, url):
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        
        return  BeautifulSoup(response.content, 'html.parser')
    
    def scrape_article_content(self, newsapi: NewsApiJsonResponseArticle) -> ScrapeArticleResponse:
        """
        Simple scraper that gets body content and removes unwanted elements
        """
        scraper_response = ScrapeArticleResponse()
        scraper_response.url = newsapi.url

        try:
            # to stagger requests
            time.sleep(random.random()) 
            
            soup = self._get_soup(newsapi.url)
            response = self.session.get(newsapi.url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Try to get relevant tags
            html_content = soup.article # will grab first one (hopefully just one)
            if not html_content:
                html_content = soup.main
            if not html_content:
                html_content = soup.body
            # Get the body element
            if not html_content:
                html_content = soup  # Fallback if no body tag
            
            # Remove unwanted elements
            for tag_name in self.unwanted_tags:
                for tag in html_content.find_all(tag_name):
                    tag.decompose()  # Remove from tree completely
            
            # Extract just the text content
            content = html_content.get_text()
            
            # Basic cleanup - remove extra whitespace
            scraper_response.content = ' '.join(content.split())
            scraper_response.word_count = len(scraper_response.content.split())
            scraper_response.success = True

            return scraper_response
            
        except Exception as e:
            scraper_response.error = str(e)
            # content defaults to None and success defaults to False
            return scraper_response

    def scrape_other_content(self, article_url: str) -> ScrapeArticleResponse:
        """
        Simple scraper that gets body content and removes unwanted elements
        """
        scraper_response = ScrapeArticleResponse()
        scraper_response.url = article_url

        try:
            # to stagger requests
            time.sleep(random.random()) 
            
            soup = self._get_soup(article_url)
            response = self.session.get(article_url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Try to get relevant tags
            html_content = soup.article # will grab first one (hopefully just one)
            if not html_content:
                html_content = soup.main
            if not html_content:
                html_content = soup.body
            # Get the body element
            if not html_content:
                html_content = soup  # Fallback if no body tag
            
            # Remove unwanted elements
            for tag_name in self.unwanted_tags:
                for tag in html_content.find_all(tag_name):
                    tag.decompose()  # Remove from tree completely
            
            # Extract just the text content
            content = html_content.get_text()
            
            # Basic cleanup - remove extra whitespace
            scraper_response.content = ' '.join(content.split())
            scraper_response.word_count = len(scraper_response.content.split())
            scraper_response.success = True

            return scraper_response
            
        except Exception as e:
            scraper_response.error = str(e)
            # content defaults to None and success defaults to False
            return scraper_response
    
    
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

# GDELT Project - Has Rate Limit

class GDELTClient:
    def __init__(self):
        self.base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
        
    def search_news(self, keywords, start_date=None, end_date=None, max_records=100, 
        language="english", sort_by="hybridrel"):
        """
        Search GDELT for news articles
        
        Args:
            keywords: List of keywords or single string
            start_date: Start date as string "YYYYMMDD" or datetime object
            end_date: End date as string "YYYYMMDD" or datetime object  
            max_records: Max articles to return (1-250)
            language: "english", "spanish", etc.
            sort_by: "hybridrel" (relevance), "datedesc" (newest first), "dateasc" (oldest first)
        """
        
        ## This creates issues!
        # Handle keywords
        # if isinstance(keywords, list):
        #     query = "+".join(keywords)
        # else:
        #     query = keywords
            
        # Handle dates
        if start_date is None:
            # Default to 30 days ago
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
        elif isinstance(start_date, datetime):
            start_date = start_date.strftime("%Y%m%d")
            
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        elif isinstance(end_date, datetime):
            end_date = end_date.strftime("%Y%m%d")

        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
            "Accept": "*/*",
        }
            
        # If EndDateTime is blank it checks last 3 months - good enough for now. 
        # Build parameters
        params = {
            'query': keywords,
            'mode': 'artlist',
            'maxrecords': min(max_records, 250),  # GDELT max is 250
            'format': 'json',
            # 'startdatetime': start_date + "000000",
            # 'enddatetime': end_date + "235959",
            'sort': sort_by
        }

        # There is also 'searchbycountry' = US but skip for now
        
        # Add language filter
        if language and language.lower() != "all":
            params['sourcelang'] = language.lower()
            
        try:
            print(f"Searching GDELT for: '{keywords}'")
            
            response = requests.get(self.base_url, headers=headers, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            # if 'articles' not in data:
            #     print("No articles found in GDELT response")
            #     return []
                
            articles = data.get('articles', [])
            if len(articles) == 0:
                print("No articles found in GDELT response")
                return []
            print(f"Found {len(articles)} articles")
            
            # Clean and structure the results
            cleaned_articles = []
            for article in articles:
                # based on response from POSTMAN
                cleaned_article = {
                    'title': article.get('title', ''),
                    'url': article.get('url', ''),
                    'url_mobile': article.get('url_mobile', ''),
                    'source': article.get('domain', ''),
                    'published_at': self._parse_gdelt_date(article.get('seendate', '')),
                    'language': article.get('language', ''),
                    'word_count': article.get('wordcount', 0),
                    'social_shares': article.get('socialimage', 0)
                }
                cleaned_articles.append(cleaned_article)
                
            return cleaned_articles
            
        except requests.exceptions.RequestException as e:
            print(f"Error making GDELT request: {e}")
            return []
        except json.JSONDecodeError as e:
            print(f"Error parsing GDELT response: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error: {e}")
            return []
    
    def search_by_domain(self, keywords, domains, start_date=None, end_date=None, max_records=50):
        """Search specific news domains"""
        if isinstance(domains, list):
            domain_filter = " OR ".join([f"domain:{domain}" for domain in domains])
        else:
            domain_filter = f"domain:{domains}"
            
        if isinstance(keywords, list):
            keyword_query = " ".join(keywords)
        else:
            keyword_query = keywords
            
        full_query = f"({keyword_query}) AND ({domain_filter})"
        
        return self.search_news(
            keywords=full_query,
            start_date=start_date,
            end_date=end_date,
            max_records=max_records
        )
    
    def _parse_gdelt_date(self, gdelt_date):
        """Parse GDELT date format to readable format"""
        if not gdelt_date or len(gdelt_date) < 8:
            return ""
            
        try:
            # GDELT date format is usually YYYYMMDDHHMMSS
            date_part = gdelt_date[:8]  # Take just YYYYMMDD
            parsed = datetime.strptime(date_part, "%Y%m%d")
            return parsed.strftime("%Y-%m-%d")
        except:
            return gdelt_date
    
    def get_trending_topics(self, hours_back=24):
        """Get trending topics from GDELT (different API endpoint)"""
        trend_url = "https://api.gdeltproject.org/api/v2/summary/summary"
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)
        
        params = {
            'startdatetime': start_time.strftime("%Y%m%d%H%M%S"),
            'enddatetime': end_time.strftime("%Y%m%d%H%M%S"),
            'format': 'json'
        }
        
        try:
            response = requests.get(trend_url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error getting trending topics: {e}")
            return {}
        

# Should use a Session
class WikiClient:
    def __init__(self):
        if os.getenv("WIKI_MEDIA_USER_AGENT") is None:
            read_dot_env()
        self.lang = 'en'
        self.user_agent = os.getenv("WIKI_MEDIA_USER_AGENT")

    @staticmethod
    def extract_text_from_html(html_content: str) -> str:
        """
        Extract clean text from Wikipedia HTML content using BeautifulSoup.
        """
        if not html_content:
            return ""
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'sup', 'table']):
            element.decompose()
        
        # Get text and clean it up
        text = soup.get_text()
        
        # Clean up whitespace and newlines
        text = re.sub(r'\n\s*\n', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
        
    def fetch(self, page: str):
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "*/*",
        }

        url = url_builder('https', [self.lang], 'wikipedia', 'org', ['w', 'api.php'])
        params = {
            'action': 'query', 
            "format": "json",
            "titles": page,
            "prop": "extracts", # Gets the HTML in { "parse": {"title": ..., "text": { "*": <html> }}}
        }
        datum = None
        try:
            response = requests.get(url, headers=headers, params=params)
            print(f"HTTP Status {response.status_code}")
            if response.status_code > 299:
                return {
                    'success': False
                }
            else:
                datum = response.json()
        except:
            return {
                'success': False
            }
        
        data_query = datum.get('query')
        if data_query is None:
            return {
                'success': False
            }
        data_pages = data_query.get('pages')
        store = None
        if data_pages is not None:
            for key, val in data_pages.items():
                # shoule be just one page for now because just one title...
                if store is None:
                    store = {
                        'title': val.get('title'),
                        'pageid': val.get('pageid'),
                        'extract': val.get('extract'),
                    }


        if store is None:
            return {
                'success': False
            }
        else:
            source = url_builder('https', [self.lang], 'wikipedia', 'org', ['wiki', page])
            return {
                'success': True,
                'title': store.get('title'),
                'url': source,
                'pageid': store.get('pageid'),
                'html': store.get('extract')
            }

    
    def search(self, query: List[str]) -> str:
        user = self.user_agent if self.user_agent is not None else "Researcher (researcher@example.com)"
        number_of_results = 1
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "*/*",
        }
        url = url_builder('https', ['api'], 'wikimedia', 'org', ['core', 'v1', 'wikipedia', 'en', 'search', 'page'])

        page_path = None
        try:
            for q in [5, 3]:
                search_query = " ".join(query[:q])
                params = {'q': search_query, 'limit': number_of_results}
                response = requests.get(url, headers=headers, params=params)

                # OK
                if response.status_code < 300:
                    pages = response.json().get('pages')
                    if pages is not None and len(pages) > 0:
                        page_path = pages[0].get('key')
                    else:
                        continue
                else:
                    # Some issue with wiki
                    continue # try with less keywords
        except:
            return {
                'success': False
            }

        return {
            'success': True,
            'page_path': page_path
        }
    
    def __call__(self, query: List[str], exists: Callable[[str], bool]) -> Dict[str, Any]:
        search_result = self.search(query)

        bad_result = { 'success': False }
        
        if not search_result.get('success'):
            return bad_result
        
        page_path = search_result.get('page_path')

        if page_path is None:
            return bad_result
        
        # Check before fetching
        source = url_builder('https', [self.lang], 'wikipedia', 'org', ['wiki', page_path])
        if exists(source):
            # technically shouldn't return failure - but OK for now
            return bad_result
        
        fetch_result = self.fetch(page_path)

        if not fetch_result.get('success'):
            return bad_result

        html = fetch_result.get('html')

        if html is None:
            return bad_result

        text_content = self.extract_text_from_html(html)

        fetch_result['text_content'] = text_content

        return fetch_result
