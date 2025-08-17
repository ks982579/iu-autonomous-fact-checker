import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any
import hashlib

class VectorPipeline:
    def __init__(self, persist_directory="./chroma_db", chunk_size: int = 256, overlap: int = 25):
        """Initialize ChromaDB client with persistence"""
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Create or get collection for fact-checking articles
        # This should be thread safe
        self.collection = self.client.get_or_create_collection(
            name="fact_check_articles",
            metadata={"description": "Chunks of news articles for fact-checking"}
        )
        self.chunk_size = chunk_size
        self.overlap = overlap
    
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