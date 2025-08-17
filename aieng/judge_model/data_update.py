import requests
import os
import json
from typing import Any, List, Optional, Dict
import base64
from pathlib import Path
import pandas as pd
import shutil
import requests
import chromadb
from bs4 import BeautifulSoup
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import hashlib

def url_builder(protocol: str, subdomains: Optional[List[str]], second_level_domain: str, top_level_domain: str, sub_directories: Optional[List[str]]):
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


def get_old_training_data():
    data_path = Path('../') / 'judge_model' / ".datasets" / "processed" / "fever_train_chunk_0000.parquet"
    print(data_path.exists())
    dft = pd.read_parquet(data_path)
    return dft

def pull_titles_from_original_data(jsonl_file_path: Path, tag: str):
    """
    This will make new parquet file for titles
    """
    output_dir_path = Path('./') / ".datasets" / "updated" # / "fever_train_chunk_0000.parquet"
 
    # def process_data(self, tag: str,  jsonl_file_path: Path, output_dir_path: Path, chunk_size: int = 1000):
    processed_data = []
    chunk_size = 500000
    chunk_num = 0
    total_processed = 0
    output_dir_path.mkdir(parents=True, exist_ok=True)
    assert jsonl_file_path.exists()
    # with sqlite3.connect(self.wikidbpath) as conn: # Database Context Manager
    line_count = sum(1 for _ in open(jsonl_file_path, 'rb'))
    with open(jsonl_file_path, 'r', encoding='utf-8') as file: # File Context Manager
        cn = 0
        size = shutil.get_terminal_size()
        loading_bar = size.columns > 20
        bar_len = min(40, size.columns-10)
        loading_text = None
        backspace = 0
        
        # READING FILE
        for line in file.readlines():

            # progress bar
            progress = int(round((cn/line_count) * bar_len, 0))
            if loading_text is not None: 
                backspace = len(loading_text)
                print("\b"*backspace, sep="", end="") # Removing Old 
                print(" "*backspace, sep="", end="")
                print("\b"*backspace, sep="", end="")
            if loading_bar:
                loading_text = f'|{"=" * (progress)}{" " * (bar_len - progress)}| {cn/line_count:.2%}'
            else:
                loading_text = f'{1/line_count:.2%}'
            print(loading_text, sep="", end="", flush=True)
            cn += 1
            # PROGRESS BAR is OVER

            cache = {}
            if len(line.strip()) == 0:
                continue
            datum = json.loads(line)
            challenge = datum.get("challenge", '').strip().casefold()
            
            # skipping numerical reasoning challenges
            if challenge == '' or challenge == "numerical reasoning":
                continue

            claim_id = datum.get('id')
            # claim = datum.get('claim')
            # label = datum.get('label')
            # support = []
            titles = []
            pages = []

            evidence = datum.get("evidence") # a list of { content, context }
            # Evidence can have multiple contents...
            for pair in evidence:

                content = pair.get("content") # list of strings
                if content is not None and len(content) > 0:
                    line = content[0]
                    parts = line.split("_") # [title, ...rest]
                    title = parts[0].strip()
                    if title not in titles:
                        titles.append(title)
                        pages.append(title.replace(" ", "_"))

            # We now have our pieces of data
            if len(titles) > 0:
                processed_data.append({
                    'id': claim_id,
                    'titles': titles,
                    'pages': pages,
                })
                total_processed += 1
            
            if len(processed_data) >= chunk_size:
                # SAVE CHUNK into Parquet
                df = pd.DataFrame(processed_data)
                chunk_file = output_dir_path / f"fever_{tag}_chunk_{chunk_num:04d}.parquet"
                df.to_parquet(chunk_file, index=False)
                ## increment chunk
                chunk_num += 1
        # End of Loop
        
        # Last Chunk
        if len(processed_data) > 0:
            # SAVE CHUNK into Parquet
            df = pd.DataFrame(processed_data)
            chunk_file = output_dir_path / f"fever_{tag}_chunk_{chunk_num:04d}.parquet"
            df.to_parquet(chunk_file, index=False)
            ## increment chunk
            chunk_num += 1
        
        # Finish Loading
        if loading_text is not None: 
            backspace = len(loading_text)
            print("\b"*backspace, sep="", end="") # Removing Old 
            print(" "*backspace, sep="", end="")
            print("\b"*backspace, sep="", end="")
        if loading_bar:
            loading_text = f'|{"=" * (bar_len)}| {1:.2%}'
        else:
            loading_text = f'{1/line_count:.2%}'
        print(loading_text, sep="", end="\n", flush=True) # New Line


def merge_datasets(dataset_type: str = 'train'):
    """
    Merge processed and updated datasets for training or dev sets.
    
    Args:
        dataset_type: Either 'train' or 'dev'
    """
    # Define paths
    processed_path = Path('./') / ".datasets" / "processed" / f"fever_{dataset_type}_chunk_0000.parquet"
    updated_path = Path('./') / ".datasets" / "updated" / f"fever_{dataset_type}_titles_chunk_0000.parquet"
    output_dir = Path('./') / ".datasets" / "updated"
    output_path = output_dir / f"fever_{dataset_type}_merged.parquet"
    
    # Check if files exist
    if not processed_path.exists():
        print(f"Error: Processed file {processed_path} does not exist")
        return None
    
    if not updated_path.exists():
        print(f"Error: Updated file {updated_path} does not exist")
        return None
    
    # Load datasets
    print(f"Loading {dataset_type} datasets...")
    processed_df = pd.read_parquet(processed_path)
    updated_df = pd.read_parquet(updated_path)
    
    print(f"Processed data shape: {processed_df.shape}")
    print(f"Updated data shape: {updated_df.shape}")
    
    # Merge on 'id' column
    merged_df = processed_df.merge(updated_df, on='id', how='inner')
    print(f"Merged data shape: {merged_df.shape}")
    
    # Validate that all entries have titles
    before_filter = len(merged_df)
    merged_df = merged_df[merged_df['titles'].apply(lambda x: x is not None and len(x) > 0)]
    after_filter = len(merged_df)
    
    if before_filter != after_filter:
        print(f"Warning: Removed {before_filter - after_filter} entries with empty titles")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save merged dataset
    merged_df.to_parquet(output_path, index=False)
    print(f"Saved merged {dataset_type} dataset to: {output_path}")
    print(f"Final dataset shape: {merged_df.shape}")
    print(f"Columns: {list(merged_df.columns)}")
    
    return merged_df


def merge_all_datasets():
    """
    Merge both training and dev datasets.
    """
    print("=== Merging Training Datasets ===")
    train_df = merge_datasets('train')
    
    print("\n=== Merging Dev Datasets ===")
    dev_df = merge_datasets('dev')
    
    return train_df, dev_df


def create_balanced_dataset():
    """
    Create a balanced dataset with:
    - 10,000 SUPPORTS samples
    - 10,000 REFUTES samples  
    - 10,000 NOT ENOUGH INFO samples (existing + relabeled from unused SUPPORTS/REFUTES)
    """
    # Load merged training data
    merged_path = Path('./') / ".datasets" / "updated" / "fever_train_merged.parquet"
    if not merged_path.exists():
        print("Merged training data not found. Creating it first...")
        merge_datasets('train')
    
    df = pd.read_parquet(merged_path)
    print(f"Original dataset shape: {df.shape}")
    print("Original label distribution:")
    print(df['label'].value_counts())
    
    # Separate by labels
    supports_df = df[df['label'] == 'SUPPORTS'].copy()
    refutes_df = df[df['label'] == 'REFUTES'].copy()  
    not_enough_info_df = df[df['label'] == 'NOT ENOUGH INFO'].copy()
    
    # Sample 10k from SUPPORTS and REFUTES
    supports_sample = supports_df.sample(n=10000, random_state=42)
    refutes_sample = refutes_df.sample(n=10000, random_state=42)
    
    print(f"Sampled {len(supports_sample)} SUPPORTS")
    print(f"Sampled {len(refutes_sample)} REFUTES")
    
    # Get all existing NOT ENOUGH INFO
    not_enough_sample = not_enough_info_df.copy()
    print(f"Existing NOT ENOUGH INFO: {len(not_enough_sample)}")
    
    # Calculate how many more NOT ENOUGH INFO we need
    needed_not_enough = 10000 - len(not_enough_sample)
    print(f"Need {needed_not_enough} more NOT ENOUGH INFO samples")
    
    if needed_not_enough > 0:
        # Get unused SUPPORTS and REFUTES (those not in our 10k samples)
        used_ids = set(supports_sample['id'].tolist() + refutes_sample['id'].tolist())
        unused_supports = supports_df[~supports_df['id'].isin(used_ids)]
        unused_refutes = refutes_df[~refutes_df['id'].isin(used_ids)]
        
        print(f"Unused SUPPORTS: {len(unused_supports)}")
        print(f"Unused REFUTES: {len(unused_refutes)}")
        
        # Take 50/50 split from unused claims
        half_needed = needed_not_enough // 2
        remainder = needed_not_enough % 2
        
        # Sample from unused claims
        additional_from_supports = unused_supports.sample(n=half_needed, random_state=42).copy()
        additional_from_refutes = unused_refutes.sample(n=half_needed + remainder, random_state=42).copy()
        
        # Relabel as NOT ENOUGH INFO
        additional_from_supports['label'] = 'NOT ENOUGH INFO'
        additional_from_refutes['label'] = 'NOT ENOUGH INFO'
        
        # Combine all NOT ENOUGH INFO samples
        not_enough_sample = pd.concat([
            not_enough_sample,
            additional_from_supports,
            additional_from_refutes
        ], ignore_index=True)
        
        print(f"Added {len(additional_from_supports)} relabeled SUPPORTS")
        print(f"Added {len(additional_from_refutes)} relabeled REFUTES")
    
    # Combine all samples into balanced dataset
    balanced_df = pd.concat([
        supports_sample,
        refutes_sample,
        not_enough_sample
    ], ignore_index=True)
    
    # Shuffle the dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nBalanced dataset shape: {balanced_df.shape}")
    print("Balanced label distribution:")
    print(balanced_df['label'].value_counts())
    
    # Save balanced dataset
    output_dir = Path('./') / ".datasets" / "updated"
    output_path = output_dir / "fever_train_balanced.parquet"
    balanced_df.to_parquet(output_path, index=False)
    print(f"\nSaved balanced dataset to: {output_path}")
    
    return balanced_df

def url_builder(protocol: str, subdomains: Optional[List[str]], second_level_domain: str, top_level_domain: str, sub_directories: Optional[List[str]]):
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

def get_wiki_data(page: str, user_agent: str):
    language_code = 'en'
    headers = {
        # "Authorization": f"Bearer {access_token.strip()}",
        "User-Agent": user_agent,
        "Accept": "*/*",
    }

    url = url_builder('https', [language_code], 'wikipedia', 'org', ['w', 'api.php'])
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
        source = url_builder('https', [language_code], 'wikipedia', 'org', ['wiki', page])
        return {
            'success': True,
            'title': store.get('title'),
            'source_url': source,
            'pageid': store.get('pageid'),
            'html': store.get('extract')
        }


def split_balanced_dataset():
    """
    Split the balanced dataset into 4 chunks of 7500 samples each,
    maintaining balanced label distribution (2500 per label per chunk).
    """
    # Load balanced dataset
    balanced_path = Path('./') / ".datasets" / "updated" / "fever_train_balanced.parquet"
    if not balanced_path.exists():
        print("Balanced dataset not found. Creating it first...")
        create_balanced_dataset()
    
    df = pd.read_parquet(balanced_path)
    print(f"Total balanced dataset shape: {df.shape}")
    print(f"Original label distribution: {df['label'].value_counts().to_dict()}")
    
    # Separate by labels
    supports_df = df[df['label'] == 'SUPPORTS'].copy().reset_index(drop=True)
    refutes_df = df[df['label'] == 'REFUTES'].copy().reset_index(drop=True)
    not_enough_df = df[df['label'] == 'NOT ENOUGH INFO'].copy().reset_index(drop=True)
    
    # Split each label into 4 chunks of 2500 each
    chunk_size_per_label = 2500
    num_chunks = 4
    
    output_dir = Path('./') / ".datasets" / "updated"
    
    for i in range(num_chunks):
        start_idx = i * chunk_size_per_label
        end_idx = start_idx + chunk_size_per_label
        
        # Get balanced chunk from each label
        supports_chunk = supports_df.iloc[start_idx:end_idx].copy()
        refutes_chunk = refutes_df.iloc[start_idx:end_idx].copy()
        not_enough_chunk = not_enough_df.iloc[start_idx:end_idx].copy()
        
        # Combine and shuffle
        chunk_df = pd.concat([supports_chunk, refutes_chunk, not_enough_chunk], ignore_index=True)
        chunk_df = chunk_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save chunk
        chunk_file = output_dir / f"fever_train_balanced_{i+1:03d}.parquet"
        chunk_df.to_parquet(chunk_file, index=False)
        
        print(f"Saved chunk {i+1}: {chunk_file} with {len(chunk_df)} samples")
        print(f"  Label distribution: {chunk_df['label'].value_counts().to_dict()}")
    
    return True


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


def chunk_text(text: str, chunk_size: int = 64, overlap: int = 8) -> List[str]:
    """
    Split text into chunks of specified word count with overlap.
    """
    if not text:
        return []
    
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(' '.join(chunk_words))
        
        if end >= len(words):
            break
            
        start = end - overlap
    
    return chunks


def setup_chromadb():
    """
    Set up ChromaDB vector store in the chromadb/ directory.
    """
    chromadb_dir = Path('./') / "chromadb"
    chromadb_dir.mkdir(exist_ok=True)
    
    client = chromadb.PersistentClient(path=str(chromadb_dir))
    
    # Create collection for judge model evidence
    collection_name = "judge_evidence"
    try:
        collection = client.get_collection(collection_name)
        print(f"Using existing collection: {collection_name}")
    except:
        collection = client.create_collection(
            name=collection_name,
            metadata={"description": "Evidence chunks for fact-checking judge model"}
        )
        print(f"Created new collection: {collection_name}")
    
    return client, collection


def process_wikipedia_page(page: str, user_agent: str, processed_urls_lock: Lock, processed_urls: set) -> dict:
    """
    Process a single Wikipedia page and return chunks data.
    """
    # Build source URL for deduplication check
    source_url = f"https://en.wikipedia.org/wiki/{page}"
    
    # Thread-safe check for processed URLs
    with processed_urls_lock:
        if source_url in processed_urls:
            return {'skipped': True, 'page': page}
    
    print(f"  Fetching Wikipedia data for: {page}")
    
    # Get Wikipedia data
    wiki_data = get_wiki_data(page, user_agent)
    
    if not wiki_data.get('success'):
        print(f"    Failed to fetch data for {page}")
        return {'error': True, 'page': page}
    
    # Extract text from HTML
    html_content = wiki_data.get('html', '')
    text_content = extract_text_from_html(html_content)
    
    if not text_content:
        print(f"    No text content found for {page}")
        return {'error': True, 'page': page}
    
    # Chunk the text
    text_chunks = chunk_text(text_content, chunk_size=64, overlap=8)
    
    if not text_chunks:
        return {'error': True, 'page': page}
    
    # Prepare chunks data
    chunk_ids = []
    chunk_texts = []
    chunk_metadatas = []
    
    for chunk_idx, chunk_txt in enumerate(text_chunks):
        chunk_id = f"{page}_chunk_{chunk_idx:04d}"
        chunk_ids.append(chunk_id)
        chunk_texts.append(chunk_txt)
        chunk_metadatas.append({
            'source_url': wiki_data.get('source_url'),
            'title': wiki_data.get('title'),
            'page': page,
            'chunk_index': chunk_idx,
            'total_chunks': len(text_chunks)
        })
    
    # Mark as processed
    with processed_urls_lock:
        processed_urls.add(source_url)
    
    print(f"    Processed {len(chunk_ids)} chunks for {page}")
    
    # Small delay to be respectful to Wikipedia
    time.sleep(0.25)  # Slightly longer delay since we're parallelizing
    
    return {
        'success': True,
        'page': page,
        'chunk_ids': chunk_ids,
        'chunk_texts': chunk_texts,
        'chunk_metadatas': chunk_metadatas
    }


def build_vector_store_from_chunk(chunk_number: int = 1):
    """
    Build vector store from a specific chunk of the balanced dataset using parallel processing.
    """
    # Setup ChromaDB
    client, collection = setup_chromadb()
    
    # Load the chunk
    chunk_file = Path('./') / ".datasets" / "updated" / f"fever_train_balanced_{chunk_number:03d}.parquet"
    if not chunk_file.exists():
        print(f"Chunk file {chunk_file} not found. Creating chunks first...")
        split_balanced_dataset()
    
    df = pd.read_parquet(chunk_file)
    print(f"Processing chunk {chunk_number} with {len(df)} samples")
    
    # Get user agent from environment
    user_agent = os.getenv("WIKI_MEDIA_USER_AGENT")
    if not user_agent:
        user_agent = "Research/1.0 (https://example.com/contact)"
        print("Warning: Using default user agent. Set WIKI_MEDIA_USER_AGENT environment variable.")
    
    # Track processed URLs to avoid duplicates (thread-safe)
    processed_urls = set()
    processed_urls_lock = Lock()
    
    # Get existing URLs from collection to avoid re-processing
    try:
        existing_metadata = collection.get()
        if existing_metadata and 'metadatas' in existing_metadata:
            for metadata in existing_metadata['metadatas']:
                if metadata and 'source_url' in metadata:
                    processed_urls.add(metadata['source_url'])
        print(f"Found {len(processed_urls)} existing URLs in vector store")
    except:
        print("Starting with empty vector store")
    
    # Collect all unique pages from the dataset
    all_pages = set()
    for idx, row in df.iterrows():
        pages = row.get('pages', [])
        for page in pages:
            if page:
                all_pages.add(page)
    
    all_pages = list(all_pages)
    print(f"Found {len(all_pages)} unique pages to process")
    
    total_chunks_added = 0
    processed_count = 0
    
    # Process pages in parallel with 4 workers
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all tasks
        future_to_page = {
            executor.submit(process_wikipedia_page, page, user_agent, processed_urls_lock, processed_urls): page
            for page in all_pages
        }
        
        # Process completed tasks
        for future in as_completed(future_to_page):
            page = future_to_page[future]
            processed_count += 1
            
            try:
                result = future.result()
                
                if result.get('skipped'):
                    print(f"[{processed_count}/{len(all_pages)}] Skipped {page} (already processed)")
                    continue
                
                if result.get('error'):
                    print(f"[{processed_count}/{len(all_pages)}] Error processing {page}")
                    continue
                
                if result.get('success'):
                    # Add chunks to vector store
                    try:
                        collection.add(
                            ids=result['chunk_ids'],
                            documents=result['chunk_texts'],
                            metadatas=result['chunk_metadatas']
                        )
                        total_chunks_added += len(result['chunk_ids'])
                        print(f"[{processed_count}/{len(all_pages)}] Added {len(result['chunk_ids'])} chunks for {page}")
                        
                    except Exception as e:
                        print(f"[{processed_count}/{len(all_pages)}] Error adding chunks for {page}: {e}")
                        continue
                        
            except Exception as e:
                print(f"[{processed_count}/{len(all_pages)}] Exception processing {page}: {e}")
                continue
    
    print(f"\nCompleted processing chunk {chunk_number}")
    print(f"Total chunks added to vector store: {total_chunks_added}")
    print(f"Vector store now contains {collection.count()} total chunks")
    
    return collection

# Could probably just inherit from vectordb.py::VectorPipeline.
class WikiVectorPipeline:
    def __init__(self, persist_dir: Optional[Path]=None, chunk_size: int = 64, overlap: int = 8):
        self.persist_dir = Path(__file__).resolve().parent / 'chromadb' if persist_dir is None else persist_dir
        # Create if not there.
        self.persist_dir.mkdir(exist_ok=True)
        """Initialize ChromaDB client with persistence"""
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection_name = "judge_evidence"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Evidence chunks for fact-checking judge model"}
        )
        
        # Create or get collection for fact-checking articles
        # This should be thread safe
        self.collection = self.client.get_or_create_collection(
            name="fact_check_articles",
            metadata={"description": "Chunks of news articles for fact-checking"}
        )
        self.chunk_size = chunk_size
        self.overlap = overlap

    # Copied from vectordb.py
    def chunk_text(self, text: str) -> List[str]:
        """
        Simple text chunking by number of words with overlap
        ~~default size of 512 words and 50 word overlap should work well in most situations.~~
        Defaulting to 256 words and 25 word overlap for more precise lookups - less for the judge AI to review.
        """
        if not text:
            return []
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            chunks.append(chunk_text)
            
            # Break if we've reached the end
            if i + self.chunk_size >= len(words):
                break
        
        return chunks
    
    # TODO: Not the same
    def create_chunk_id(self, url: str, chunk_index: int) -> str:
        """Create a unique ID for each chunk"""
        base_string = f"{url}_{chunk_index}"
        return hashlib.md5(base_string.encode()).hexdigest()
    
    # TODO: Not the same
    def add_article(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an article: chunk it, embed it, and store in ChromaDB
        
        article_data should have: url, content, title, source, published_at, etc.
        """
        url = article_data.get('url')
        content = article_data.get('full_content') or article_data.get('content', '')
        
        if not content or not url:
            return {"success": False, "error": "Missing content or URL"}
        
        # Check if we already have this article
        if self.article_exists(url):
            return {"success": False, "error": "Article already exists", "url": url}
        
        try:
            # Chunk the content
            chunks = self.chunk_text(content)
            
            if not chunks:
                return {"success": False, "error": "No chunks created"}
            
            # Prepare data for ChromaDB
            chunk_ids = []
            chunk_texts = []
            chunk_metadata = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = self.create_chunk_id(url, i)
                
                # Wrappped in strings because some things come in as None
                # metadata can't have null values.
                metadata = {
                    "url": url,
                    "chunk_index": i,
                    "title": str(article_data.get('title', '')),
                    "source": str(article_data.get('source', '')),
                    "author": str(article_data.get('author', '')),
                    "published_at": str(article_data.get('published_at', '')),
                    "word_count": len(chunk.split())
                }
                
                chunk_ids.append(chunk_id)
                chunk_texts.append(chunk)
                chunk_metadata.append(metadata)
            
            # Add to ChromaDB (ChromaDB handles embedding automatically)
            self.collection.add(
                documents=chunk_texts,
                metadatas=chunk_metadata,
                ids=chunk_ids
            )
            
            return {
                "success": True,
                "url": url,
                "chunks_created": len(chunks),
                "total_words": len(content.split())
            }
            
        except Exception as e:
            return {"success": False, "error": str(e), "url": url, "metadata": metadata}
    
    def article_exists(self, url: str) -> bool:
        """Check if an article URL already exists in the database"""
        try:
            results = self.collection.get(
                where={"source_url": url},
                limit=1
            )
            return len(results['ids']) > 0
        except:
            return False
    
    def search_similar_content(self, query: str, n_results: int = 10) -> Dict[str, Any]:
        """
        Search for content similar to the query claim
        Returns chunks that might be relevant for fact-checking
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results for easier use
            formatted_results = []
            
            if results['ids'] and results['ids'][0]:  # Check if we got results
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        'chunk_id': results['ids'][0][i],
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'similarity_score': 1 - results['distances'][0][i]  # Convert distance to similarity
                    })
            
            return {
                "success": True,
                "query": query,
                "results": formatted_results,
                "total_found": len(formatted_results)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e), "query": query}
    
    def get_article_chunks(self, url: str) -> List[Dict]:
        """Get all chunks for a specific article URL"""
        try:
            results = self.collection.get(
                where={"url": url},
                include=['documents', 'metadatas']
            )
            
            chunks = []
            for i in range(len(results['ids'])):
                chunks.append({
                    'chunk_id': results['ids'][i],
                    'content': results['documents'][i],
                    'metadata': results['metadatas'][i]
                })
            
            # Sort by chunk_index
            chunks.sort(key=lambda x: x['metadata']['chunk_index'])
            return chunks
            
        except Exception as e:
            print(f"Error getting chunks for {url}: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the database"""
        try:
            total_chunks = self.collection.count()
            
            # Get unique URLs
            all_metadata = self.collection.get(include=['metadatas'])
            unique_urls = set()
            if all_metadata['metadatas']:
                unique_urls = set(metadata['url'] for metadata in all_metadata['metadatas'])
            
            return {
                "total_chunks": total_chunks,
                "unique_articles": len(unique_urls),
                "avg_chunks_per_article": total_chunks / len(unique_urls) if unique_urls else 0
            }
        except Exception as e:
            return {"error": str(e)}

    # Group chunks by URL and combine adjacent ones
    def group_and_combine_chunks(self, results): # TODO: Similarity Results should have proper return type to put in here
        """
        Group chunks by URL and combine adjacent chunks together.
        Returns a list of combined content blocks.


  "similarity_search_results": {
    "success": true,
    "query": "\u201cAs such, the President and we are disbanding the Forum,\u201d the group said.",
    "results": [
      {
        "chunk_id": "3b0c766fb9136ac5374478bc71db644c",
        "content": "justice for everyone. Everyone,\" Khalil said. \"Other people would see it very differently, though, right?\" Brennan asked.\"No because people interpret what they want to do,\" Khalil said. He added, \"Intifada is simply an uprising, and to globalize the Intifada is to globalize the solidarity, the uprising against injustices around the world.\" As for the future, Khalil says he will be prioritizing his family, but continue protesting.\"When you look back over the past couple of years, do you ever say to yourself, ah, I might have done this differently?\" Brennan asked. \"Absolutely. I could have communicated better, built more bridges, but the core thing, which is opposing a genocide, opposing a war, I wouldn't have changed that,\" Khalil said. Featured Stories & Web Exclusives Dick Brennan Dick Brennan joined CBS News New York in 2012 as an anchor and reporter. Featured Local Savings",
        "metadata": {
          "chunk_index": 3,
          "author": "Dick  Brennan",
          "source": "CBS News",
          "url": "https://www.cbsnews.com/newyork/news/mahmoud-khalil-columbia-protests-trump-lawsuit/",
          "published_at": "2025-07-12T04:27:00Z",
          "title": "Mahmoud Khalil discusses Trump administration lawsuit, Columbia protests",
          "word_count": 142
        },
        "similarity_score": -0.4848473072052002
      },
        """
        if not results or not results.get("results"):
            return []
        
        # Group by URL
        url_groups = {}
        for result in results["results"]:
            url = result["metadata"].get("url", "")
            if url not in url_groups:
                url_groups[url] = []
            url_groups[url].append(result) # { id, content, metadata, similarity_score}
        
        combined_results = []
        
        # Process each URL group
        # chunks: [{ id, content, metadata, similarity_score}, ...]
        for url, chunks in url_groups.items():
            # Sort chunks by chunk_index
            chunks.sort(key=lambda x: x["metadata"].get("chunk_index", 0))
            
            # Group adjacent chunks
            adjacent_groups = []
            current_group = [chunks[0]]
            
            for i in range(1, len(chunks)):
                current_chunk_index = chunks[i]["metadata"].get("chunk_index", 0)
                prev_chunk_index = chunks[i-1]["metadata"].get("chunk_index", 0)
                
                # If chunks are adjacent (difference of 1), add to current group
                if current_chunk_index == prev_chunk_index + 1:
                    current_group.append(chunks[i])
                else:
                    # Start a new group
                    adjacent_groups.append(current_group)
                    current_group = [chunks[i]]
            
            # Add the last group
            adjacent_groups.append(current_group)
            
            # Combine content for each adjacent group
            for group in adjacent_groups:
                combined_content = ""
                for i in range(0, len(group)):
                    this_chunk = group[i].get('content', '')
                    # if first chunk use the whole text
                    if i == 0:
                        combined_content += this_chunk
                    else: #Adjust for overlap
                        combined_content += this_chunk[self.overlap:]

                # Use metadata from the first chunk, but update chunk_index info
                first_chunk = group[0]
                last_chunk = group[-1]
                combined_metadata = first_chunk["metadata"].copy()
                combined_metadata["chunk_index_range"] = f"{first_chunk['metadata'].get('chunk_index', 0)}-{last_chunk['metadata'].get('chunk_index', 0)}"
                combined_metadata["chunks_combined"] = len(group)
                
                combined_results.append({
                    "content": combined_content,
                    "metadata": combined_metadata,
                    "similarity_score": max([chunk.get("similarity_score", 0) for chunk in group]),
                    "original_chunks": len(group)
                })
        # Sort by similarity score (highest first)
        combined_results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        
        return combined_results

def vector_checks(wvp: WikiVectorPipeline, chunk_number: int = 1):
    """
    Build vector store from a specific chunk of the balanced dataset using parallel processing.
    """
    # Setup ChromaDB
    
    # Load the chunk
    chunk_file = Path('./') / ".datasets" / "updated" / f"fever_train_balanced_{chunk_number:03d}.parquet"
    
    df = pd.read_parquet(chunk_file)
    print(f"Processing chunk {chunk_number} with {len(df)} samples")

    cnt = 0
    for index, row in df.iterrows():
        print(row['claim'])
        test = wvp.search_similar_content("Hello", 5)
        print(test)

        print("Search by Claim")
        res = wvp.search_similar_content(row['claim'], 5)
        print(res)
        print("---------------")

        print("Search by Evidence")
        for ev in row['evidence']:
            # print(ev)
            eres = wvp.search_similar_content(ev, 5)
            print(eres)
        print("---------------")
        break


# I've processed the first chunk
# There were some errors so probably worth processing again.
# Until then - continue.

if __name__ == "__main__":
    # jsonl_file_path = Path('./') / ".datasets" / "original" / "feverous_train_challenges.jsonl"
    # tag = "train_titles"
    # jsonl_file_path = Path('./') / ".datasets" / "original" / "feverous_dev_challenges.jsonl"
    # tag = "dev_titles"
    # pull_titles_from_original_data(jsonl_file_path, tag)
    # merge_all_datasets()
    # create_balanced_dataset()
    
    # Split balanced dataset into chunks
    # split_balanced_dataset()
    
    # Build vector store from first chunk
    # TODO: Process other chunks
    # build_vector_store_from_chunk(1)
    wiki = WikiVectorPipeline()
    vector_checks(wiki, 1)