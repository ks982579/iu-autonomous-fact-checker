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

# Could probably just inherit from vectordb.py::VectorPipeline.
class WikiVectorPipeline:
    def __init__(self, persist_dir: Optional[Path]=None, chunk_size: int = 64, overlap: int = 8):
        self.persist_dir = Path(__file__).resolve().parent / 'chromadb' if persist_dir is None else persist_dir
        # Create if not there.
        self.persist_dir.mkdir(exist_ok=True)
        """Initialize ChromaDB client with persistence"""
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection_name = "judge_evidence"  # Use the same collection name as data_update.py
        
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            print(f"Connected to existing collection: {self.collection_name}")
        except:
            print(f"Collection {self.collection_name} not found. Creating new one...")
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Evidence chunks for fact-checking judge model"}
            )
        
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Print collection stats
        try:
            count = self.collection.count()
            print(f"Collection contains {count} chunks")
        except Exception as e:
            print(f"Error getting collection count: {e}")

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
    
    # TODO: implement later inherit
    def create_chunk_id(self, url: str, chunk_index: int) -> str:
        ...
    
    # TODO: implement later inherit
    def add_article(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        ...
    
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
    
    def get_sample_data(self, limit: int = 5):
        """Get a sample of data from the collection to inspect structure"""
        try:
            results = self.collection.get(
                limit=limit,
                include=['documents', 'metadatas']
            )
            return results
        except Exception as e:
            print(f"Error getting sample data: {e}")
            return None

    def search_similar_content(self, query: str, n_results: int = 10) -> Dict[str, Any]:
        """
        Search for content similar to the query claim
        Returns chunks that might be relevant for fact-checking
        """
        try:
            # First check if collection has any data
            total_count = self.collection.count()
            if total_count == 0:
                return {
                    "success": False,
                    "error": "Collection is empty",
                    "query": query,
                    "total_count": total_count
                }
            
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, total_count),  # Don't request more than available
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
                "total_found": len(formatted_results),
                "collection_total": total_count
            }
            
        except Exception as e:
            return {"success": False, "error": str(e), "query": query}

    def search_comprehensive_evidence(self, claim: str, evidence_list: List[str], n_results_per_query: int = 10) -> Dict[str, Any]:
        """
        Search for evidence using both the claim and all evidence sentences.
        Compiles all results and sorts by similarity score.
        
        Args:
            claim: The main claim to fact-check
            evidence_list: List of evidence sentences
            n_results_per_query: Number of results to get per search query
            
        Returns:
            Dictionary with compiled and sorted results
        """
        all_results = []
        search_queries = [claim] + evidence_list
        
        print(f"Searching with {len(search_queries)} queries (claim + {len(evidence_list)} evidence):")
        
        for i, query in enumerate(search_queries):
            query_type = "claim" if i == 0 else f"evidence_{i}"
            query_preview = query[:80] + "..." if len(query) > 80 else query
            print(f"  {query_type}: {query_preview}")
            
            search_result = self.search_similar_content(query, n_results_per_query)
            
            if search_result.get('success') and search_result.get('results'):
                for result in search_result['results']:
                    # Add query information to each result
                    result['query_type'] = query_type
                    result['query_text'] = query
                    all_results.append(result)
                print(f"    → Found {len(search_result['results'])} results")
            else:
                error_msg = search_result.get('error', 'Unknown error') if search_result else 'No response'
                print(f"    → No results: {error_msg}")
        
        # Remove duplicates based on chunk_id
        unique_results = {}
        for result in all_results:
            chunk_id = result['chunk_id']
            if chunk_id not in unique_results or result['similarity_score'] > unique_results[chunk_id]['similarity_score']:
                # Keep the result with higher similarity score if duplicate
                unique_results[chunk_id] = result
        
        # Convert back to list and sort by similarity score (highest first)
        final_results = list(unique_results.values())
        final_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Group results by source page for easier analysis
        results_by_page = {}
        for result in final_results:
            page = result['metadata'].get('page', 'unknown')
            if page not in results_by_page:
                results_by_page[page] = []
            results_by_page[page].append(result)
        
        return {
            "success": True,
            "claim": claim,
            "evidence_count": len(evidence_list),
            "total_queries": len(search_queries),
            "total_results_found": len(final_results),
            "unique_pages": len(results_by_page),
            "results": final_results,
            "results_by_page": results_by_page,
            "top_similarity_score": final_results[0]['similarity_score'] if final_results else 0
        }

    def select_evidence_for_label(self, search_results: Dict[str, Any], label: str) -> List[Dict[str, Any]]:
        """
        Select appropriate evidence chunks based on the label.
        
        Args:
            search_results: Results from search_comprehensive_evidence
            label: "SUPPORTS", "REFUTES", or "NOT ENOUGH INFO"
            
        Returns:
            List of 5 selected evidence chunks
        """
        if not search_results.get('success') or not search_results.get('results'):
            return []
        
        all_results = search_results['results']
        
        if label in ["SUPPORTS", "REFUTES"]:
            # Get top 5 results with similarity > 0
            positive_results = [r for r in all_results if r['similarity_score'] > 0]
            return positive_results[:5]
        
        elif label == "NOT ENOUGH INFO":
            # Get bottom 5 results (lowest similarity scores)
            return all_results[-5:] if len(all_results) >= 5 else all_results
        
        else:
            # Fallback: return top 5
            return all_results[:5]


def build_vectorized_dataset(wvp: WikiVectorPipeline, chunk_number: int = 1):
    """
    Build the vectorized dataset with evidence from ChromaDB searches.
    
    Args:
        wvp: WikiVectorPipeline instance
        chunk_number: Which chunk to process (1-4)
    """
    # Load the chunk
    chunk_file = Path('./') / ".datasets" / "updated" / f"fever_train_balanced_{chunk_number:03d}.parquet"
    if not chunk_file.exists():
        print(f"Chunk file {chunk_file} not found!")
        return None
    
    df = pd.read_parquet(chunk_file)
    print(f"Processing chunk {chunk_number} with {len(df)} samples")
    
    # Create output directory
    output_dir = Path('./') / ".datasets" / "vectorized"
    output_dir.mkdir(exist_ok=True)
    
    # Track progress and results
    vectorized_data = []
    label_counts = {"SUPPORTS": 0, "REFUTES": 0, "NOT ENOUGH INFO": 0}
    evidence_stats = {"total_searches": 0, "successful_searches": 0, "failed_searches": 0}
    
    for idx, row in df.iterrows():
        print(f"\nProcessing row {idx + 1}/{len(df)}: {row['label']}")
        
        # Get comprehensive search results
        comprehensive_results = wvp.search_comprehensive_evidence(
            claim=row['claim'], 
            evidence_list=row['evidence'], 
            n_results_per_query=10  # Get more results to have better selection
        )
        
        evidence_stats["total_searches"] += 1
        
        if comprehensive_results.get('success'):
            evidence_stats["successful_searches"] += 1
            
            # Select appropriate evidence based on label
            selected_evidence = wvp.select_evidence_for_label(comprehensive_results, row['label'])
            
            print(f"  Found {comprehensive_results.get('total_results_found', 0)} total results")
            print(f"  Selected {len(selected_evidence)} evidence chunks for label '{row['label']}'")
            
            if selected_evidence:
                similarity_scores = [e['similarity_score'] for e in selected_evidence]
                print(f"  Similarity range: {min(similarity_scores):.3f} to {max(similarity_scores):.3f}")
            
        else:
            evidence_stats["failed_searches"] += 1
            selected_evidence = []
            print(f"  Search failed: {comprehensive_results.get('error', 'Unknown error')}")
        
        # Add to vectorized data
        vectorized_row = row.copy()
        vectorized_row['vector_evidence'] = selected_evidence
        vectorized_row['evidence_count'] = len(selected_evidence)
        vectorized_data.append(vectorized_row)
        
        label_counts[row['label']] += 1
        
        # Progress update every 100 rows
        if (idx + 1) % 100 == 0:
            print(f"Progress: {idx + 1}/{len(df)} ({(idx + 1)/len(df)*100:.1f}%)")
    
    # Create final DataFrame
    vectorized_df = pd.DataFrame(vectorized_data)
    
    # Save to file
    output_file = output_dir / f"fever_train_balanced_{chunk_number:03d}.parquet"
    vectorized_df.to_parquet(output_file, index=False)
    
    # Print summary statistics
    print(f"\n=== Dataset Creation Complete ===")
    print(f"Output file: {output_file}")
    print(f"Total samples: {len(vectorized_df)}")
    print(f"Label distribution: {label_counts}")
    print(f"Search statistics: {evidence_stats}")
    print(f"Success rate: {evidence_stats['successful_searches']/evidence_stats['total_searches']*100:.1f}%")
    
    # Analyze evidence distribution
    evidence_counts = vectorized_df['evidence_count'].value_counts().sort_index()
    print(f"Evidence count distribution: {evidence_counts.to_dict()}")
    
    # Sample check
    print(f"\n=== Sample Data ===")
    for label in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]:
        sample = vectorized_df[vectorized_df['label'] == label].iloc[0]
        evidence = sample['vector_evidence']
        if evidence:
            scores = [e['similarity_score'] for e in evidence]
            print(f"{label}: {len(evidence)} chunks, similarity range: {min(scores):.3f} to {max(scores):.3f}")
        else:
            print(f"{label}: No evidence found")
    
    return vectorized_df


def vector_checks(wvp: WikiVectorPipeline, chunk_number: int = 1):
    """
    Dealing with an _issue_ with ChromaDB.
    Turns out I had an error in the Collection Name...
    Determine which is better, searching by Claim or Evidence

    Args:
        wvp (WikiVectorPipeline): _description_
        chunk_number (int, optional): _description_. Defaults to 1.
    """
    # First, inspect the collection
    print("=== Collection Inspection ===")
    sample_data = wvp.get_sample_data(3)
    if sample_data:
        print(f"Sample IDs: {sample_data.get('ids', [])[:3]}")
        print(f"Sample metadata keys: {list(sample_data.get('metadatas', [{}])[0].keys()) if sample_data.get('metadatas') else 'No metadata'}")
        if sample_data.get('metadatas'):
            print(f"First metadata: {sample_data['metadatas'][0]}")
        if sample_data.get('documents'):
            print(f"First document (first 200 chars): {sample_data['documents'][0][:200]}...")
    else:
        print("No sample data available")
    
    # Load the chunk
    chunk_file = Path('./') / ".datasets" / "updated" / f"fever_train_balanced_{chunk_number:03d}.parquet"
    
    df = pd.read_parquet(chunk_file)
    print(f"\n=== Processing chunk {chunk_number} with {len(df)} samples ===")
    
    # Test basic search first
    print("\n=== Basic Search Test ===")
    basic_test = wvp.search_similar_content("Wikipedia", 3)
    print(f"Basic search result: {basic_test}")

    cnt = 0
    for index, row in df.iterrows():
        print(f"\n=== Row {index} ===")
        print(f"Claim: {row['claim']}")
        print(f"Pages: {row['pages']}")
        print(f"Label: {row['label']}")

        # Check if any pages from this row exist in the vector store
        print(f"\n--- Page Existence Check ---")
        for page in row.get('pages', [])[:3]:  # Check first 3 pages
            page_url = f"https://en.wikipedia.org/wiki/{page}"
            exists = wvp.article_exists(page_url)
            print(f"  {page}: {exists}")
        
        print(f"\n--- Evidence List ---")
        evidence_list = row['evidence']
        print(f"Evidence sentences ({len(evidence_list)}):")
        for i, ev in enumerate(evidence_list):
            print(f"  {i+1}: {ev}")
        
        print(f"\n--- Comprehensive Evidence Search ---")
        comprehensive_results = wvp.search_comprehensive_evidence(
            claim=row['claim'], 
            evidence_list=evidence_list, 
            n_results_per_query=5
        )
        
        print(f"\nComprehensive Search Summary:")
        print(f"  Total queries: {comprehensive_results.get('total_queries', 0)}")
        print(f"  Total unique results: {comprehensive_results.get('total_results_found', 0)}")
        print(f"  Unique pages: {comprehensive_results.get('unique_pages', 0)}")
        print(f"  Top similarity score: {comprehensive_results.get('top_similarity_score', 0):.3f}")
        
        # Show top 5 results
        if comprehensive_results.get('results'):
            print(f"\nTop 5 Results (sorted by similarity):")
            for i, result in enumerate(comprehensive_results['results'][:5]):
                print(f"  {i+1}. Similarity: {result['similarity_score']:.3f}")
                print(f"     Query type: {result['query_type']}")
                print(f"     Page: {result['metadata'].get('title', 'N/A')}")
                print(f"     Content: {result['content'][:100]}...")
                print()
        
        # Show results grouped by page
        if comprehensive_results.get('results_by_page'):
            print(f"\nResults by Page:")
            for page, page_results in list(comprehensive_results['results_by_page'].items())[:3]:
                print(f"  {page}: {len(page_results)} chunks")
                best_score = max(r['similarity_score'] for r in page_results)
                print(f"    Best similarity: {best_score:.3f}")
        
        break  # Only process first row for now


# I've processed the first chunk
# There were some errors so probably worth processing again.
# Until then - continue.

def test_dataset_building(wvp: WikiVectorPipeline, chunk_number: int = 1, test_rows: int = 3):
    """Test the dataset building process with a small sample."""
    chunk_file = Path('./') / ".datasets" / "updated" / f"fever_train_balanced_{chunk_number:03d}.parquet"
    df = pd.read_parquet(chunk_file)
    
    print(f"Testing with first {test_rows} rows...")
    test_df = df.head(test_rows)
    
    for idx, row in test_df.iterrows():
        print(f"\n=== Test Row {idx + 1} ===")
        print(f"Claim: {row['claim']}")
        print(f"Label: {row['label']}")
        print(f"Evidence: {row['evidence']}")
        
        # Get comprehensive search results
        comprehensive_results = wvp.search_comprehensive_evidence(
            claim=row['claim'], 
            evidence_list=row['evidence'], 
            n_results_per_query=10
        )
        
        if comprehensive_results.get('success'):
            # Select appropriate evidence based on label
            selected_evidence = wvp.select_evidence_for_label(comprehensive_results, row['label'])
            
            print(f"Selected {len(selected_evidence)} evidence chunks for '{row['label']}':")
            for i, evidence in enumerate(selected_evidence[:3]):  # Show first 3
                print(f"  {i+1}. Score: {evidence['similarity_score']:.3f} | {evidence['content'][:80]}...")
        else:
            print(f"Search failed: {comprehensive_results.get('error')}")


if __name__ == "__main__":
    wiki = WikiVectorPipeline()
    
    # test_dataset_building(wiki, 1)
    # Test completed successfully, now build full dataset
    print("Building full vectorized dataset...")
    vectorized_df = build_vectorized_dataset(wiki, 1)
    
    if vectorized_df is not None:
        print("Dataset creation successful!")
    else:
        print("Dataset creation failed!")
