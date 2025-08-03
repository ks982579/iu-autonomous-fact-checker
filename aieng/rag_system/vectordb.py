import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any
import hashlib

class VectorPipeline:
    def __init__(self, persist_directory="./chroma_db"):
        """Initialize ChromaDB client with persistence"""
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Create or get collection for fact-checking articles
        # This should be thread safe
        self.collection = self.client.get_or_create_collection(
            name="fact_check_articles",
            metadata={"description": "Chunks of news articles for fact-checking"}
        )
    
    def chunk_text(self, text: str, chunk_size: int = 256, overlap: int = 25) -> List[str]:
        """
        Simple text chunking by number of words with overlap
        ~~default size of 512 words and 50 word overlap should work well in most situations.~~
        Defaulting to 256 words and 25 word overlap for more precise lookups - less for the judge AI to review.
        """
        if not text:
            return []
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            chunks.append(chunk_text)
            
            # Break if we've reached the end
            if i + chunk_size >= len(words):
                break
        
        return chunks
    
    def create_chunk_id(self, url: str, chunk_index: int) -> str:
        """Create a unique ID for each chunk"""
        base_string = f"{url}_{chunk_index}"
        return hashlib.md5(base_string.encode()).hexdigest()
    
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
                where={"url": url},
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

# Test usage
if __name__ == "__main__":
    # Initialize the pipeline
    pipeline = VectorPipeline()
    
    # Example article data (from your scraper)
    sample_article = {
        # Article from other example
        'url': 'https://gizmodo.com/trump-declares-musk-a-train-wreck-after-party-launch-2000624726',
        'title': 'Trump Declares Musk a “Train Wreck” After Party Launch',
        'source': 'Gizmodo.com',
        'author': 'Luc Olinga',
        'published_at': '2025-07-07',
        'full_content': 'The friendship that once saw Elon Musk serve in the White House is officially and explosively over. In a lengthy and personal tirade posted to Truth Social on Sunday, President Donald Trump unloaded … [+3390 chars]'
    }
    
    # Add article to vector database
    result = pipeline.add_article(sample_article)
    print("Add result:", result)
    
    # Search for similar content
    search_results = pipeline.search_similar_content("Truth Social post about Elon Musk.")
    print("Search results:", search_results)
    
    # Get database stats
    stats = pipeline.get_stats()
    print("Database stats:", stats)